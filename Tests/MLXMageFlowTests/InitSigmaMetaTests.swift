import XCTest
import MLXToolKit
@testable import MLXMageFlow

/// `metaData["initSigma"]` (G-5 creative upscale): validated before any model work.
final class InitSigmaMetaTests: XCTestCase {
    func testValuesPassThrough() throws {
        XCTAssertNil(try MageFlowMetaKeys.initSigma([:]))
        XCTAssertNil(try MageFlowMetaKeys.initSigma([MageFlowMetaKeys.initSigmaKey: .null]))
        XCTAssertEqual(try MageFlowMetaKeys.initSigma([MageFlowMetaKeys.initSigmaKey: .double(0.9)]), 0.9)
        XCTAssertEqual(try MageFlowMetaKeys.initSigma([MageFlowMetaKeys.initSigmaKey: .int(1)]), 1)
    }

    func testOutOfRangeOrWrongTypeIsRejected() {
        for bad: MetaValue in [.double(0), .double(-0.2), .double(1.5), .string("0.9"), .bool(true)] {
            XCTAssertThrowsError(try MageFlowMetaKeys.initSigma([MageFlowMetaKeys.initSigmaKey: bad]))
        }
    }
}
