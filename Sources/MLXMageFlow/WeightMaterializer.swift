// WeightMaterializer.swift — first-run download of the package's declared weight sources
// (v0.19.0 contract: the PACKAGE auto-materializes; the app only picks the models folder).
//
// Executes the `WeightSourcing` declaration on MageFlowConfiguration into the engine
// ModelStore layout (`<root>/<org>/<name>/…`), forwarding BYTE-ACCURATE progress to
// `WeightDownloadProgress` so the engine's PreparationMonitor surfaces a real, moving
// `.downloading(fraction:bytesPerSecond:)` phase.
//
// Files are streamed DIRECTLY to the store via URLSession — deliberately not
// HubClient.downloadSnapshot, which (as of swift-huggingface 0.9.0) has two problems
// this replaces:
//   1. its per-file Progress never receives byte updates during a transfer, so the
//      reported fraction sits at 0% for the whole duration of a large file;
//   2. it downloads into its own Python-compatible cache and COPIES to the destination,
//      double-storing every multi-GB artifact.
// HubClient is still used for the tree listing (auth + endpoint handling).

import Foundation
import HuggingFace
import MLXToolKit

public enum WeightMaterializer {

    enum MaterializeError: Error, LocalizedError {
        case badRepoId(String)
        case noStoreRoot
        case httpStatus(String, Int)
        var errorDescription: String? {
            switch self {
            case .badRepoId(let id): return "Malformed weight-source repo id '\(id)' (want org/name)."
            case .noStoreRoot:
                return "Mage-Flow has no local weights and no model store to download into — "
                    + "set an explicit snapshotPath or choose a models folder."
            case .httpStatus(let path, let code): return "Download of \(path) failed (HTTP \(code))."
            }
        }
    }

    /// Download every `source` into `root` (ModelStore layout). Progress is
    /// byte-weighted and monotonic across ALL sources' files.
    public static func materialize(_ sources: [WeightSource], into root: URL) async throws {
        let client = HubClient()   // env-detected endpoint + token; gated repos honor HF_TOKEN
        let store = ModelStore(root: root)

        // Enumerate everything first so the fraction denominator is global.
        struct Item { let repo: String; let revision: String; let path: String
                      let size: Int64; let destination: URL }
        var items: [Item] = []
        for source in sources {
            guard let repoId = Repo.ID(rawValue: source.repo),
                  let destination = store.directory(for: source.repo) else {
                throw MaterializeError.badRepoId(source.repo)
            }
            let revision = source.revision ?? "main"
            let entries = try await client.listFiles(in: repoId, revision: revision)
            for entry in entries where entry.type == .file {
                let globs = source.matching ?? []
                let matches = globs.isEmpty || globs.contains { fnmatch($0, entry.path, 0) == 0 }
                guard matches else { continue }
                let dest = destination.appendingPathComponent(entry.path)
                // Skip files already fully present (source-level resume).
                if let attrs = try? FileManager.default.attributesOfItem(atPath: dest.path),
                   (attrs[.size] as? Int64) == Int64(entry.size ?? -1) { continue }
                items.append(Item(repo: source.repo, revision: revision, path: entry.path,
                                  size: Int64(entry.size ?? 0), destination: dest))
            }
        }
        guard !items.isEmpty else { return }
        let totalBytes = max(items.reduce(0) { $0 + $1.size }, 1)

        var doneBytes: Int64 = 0
        let started = Date()
        for item in items {
            try await downloadItem(
                repo: item.repo, revision: item.revision, path: item.path, to: item.destination
            ) { fileBytes in
                let elapsed = max(Date().timeIntervalSince(started), 0.001)
                let overall = doneBytes + fileBytes
                WeightDownloadProgress.report(
                    fraction: Double(overall) / Double(totalBytes),
                    bytesPerSecond: Double(overall) / elapsed)
            }
            doneBytes += item.size
        }
        WeightDownloadProgress.report(fraction: 1.0, bytesPerSecond: nil)
    }

    /// Stream one file straight to `destination` (via a sibling .partial), counting
    /// bytes; `onBytes` is throttled to ~4 reports/second.
    private static func downloadItem(
        repo: String, revision: String, path: String, to destination: URL,
        onBytes: (Int64) -> Void
    ) async throws {
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        var request = URLRequest(url: URL(string:
            "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")!)
        if let token = hfToken() { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw MaterializeError.httpStatus(path, (response as? HTTPURLResponse)?.statusCode ?? -1)
        }

        let partial = destination.appendingPathExtension("partial")
        FileManager.default.createFile(atPath: partial.path, contents: nil)
        let handle = try FileHandle(forWritingTo: partial)

        var received: Int64 = 0
        var lastReport = Date.distantPast
        var buffer = Data(); buffer.reserveCapacity(4 << 20)
        do {
            for try await byte in bytes {
                buffer.append(byte)
                if buffer.count >= 4 << 20 {
                    try handle.write(contentsOf: buffer)
                    received += Int64(buffer.count)
                    buffer.removeAll(keepingCapacity: true)
                    let now = Date()
                    if now.timeIntervalSince(lastReport) > 0.25 {
                        lastReport = now
                        onBytes(received)
                    }
                    try Task.checkCancellation()
                }
            }
            if !buffer.isEmpty {
                try handle.write(contentsOf: buffer)
                received += Int64(buffer.count)
            }
            try handle.close()
        } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partial, to: destination)
        onBytes(received)
    }

    /// HF token: env first, then the CLI token file (upstream convention).
    private static func hfToken() -> String? {
        if let t = ProcessInfo.processInfo.environment["HF_TOKEN"], !t.isEmpty { return t }
        let file = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/token")
        return (try? String(contentsOf: file, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
