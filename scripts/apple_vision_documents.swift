// Build: xcrun swiftc -O scripts/apple_vision_documents.swift -o .build/apple-vision-documents
import Foundation
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

@available(macOS 26.0, *)
func bbox(_ region: NormalizedRegion) -> [String: Double] {
    let rect = region.boundingBox.cgRect
    // Vision uses a lower-left origin; ParseBench uses top-left coordinates.
    return ["x": rect.minX, "y": 1 - rect.maxY, "w": rect.width, "h": rect.height]
}

@available(macOS 26.0, *)
func documentJSON(_ document: DocumentObservation.Container) -> [String: Any] {
    let paragraphs: [[String: Any]] = document.paragraphs.map {
        ["text": $0.transcript, "bbox": bbox($0.boundingRegion)]
    }
    let tables: [[String: Any]] = document.tables.map { table in
        // Merged cells may occur in multiple rows. Emit each cell once.
        var seen = Set<String>()
        let cells: [[String: Any]] = table.rows.flatMap { $0 }.compactMap { cell in
            let key = "\(cell.rowRange):\(cell.columnRange)"
            guard seen.insert(key).inserted else { return nil }
            return [
                "text": cell.content.text.transcript,
                "bbox": bbox(cell.content.boundingRegion),
                "row": cell.rowRange.lowerBound,
                "column": cell.columnRange.lowerBound,
                "row_span": cell.rowRange.count,
                "column_span": cell.columnRange.count,
            ]
        }
        return [
            "bbox": bbox(table.boundingRegion), "cells": cells,
            "row_count": table.rows.count, "column_count": table.columns.count,
        ]
    }
    let lists: [[String: Any]] = document.lists.map { list in
        ["bbox": bbox(list.boundingRegion), "items": list.items.map { item in
            ["text": item.itemString, "marker": item.markerString,
             "bbox": bbox(item.content.boundingRegion)] as [String: Any]
        }]
    }
    var result: [String: Any] = [
        "text": document.text.transcript, "bbox": bbox(document.boundingRegion),
        "paragraphs": paragraphs, "tables": tables, "lists": lists,
    ]
    if let title = document.title {
        result["title"] = ["text": title.transcript, "bbox": bbox(title.boundingRegion)]
    }
    return result
}

guard CommandLine.arguments.count == 2 else {
    fail("Usage: apple-vision-documents IMAGE_PATH")
}
guard #available(macOS 26.0, *) else {
    fail("Apple Vision document recognition requires macOS 26 or newer.")
}
let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard FileManager.default.isReadableFile(atPath: imageURL.path) else {
    fail("Image is not readable: \(imageURL.path)")
}
do {
    let request = RecognizeDocumentsRequest(.revision1)
    let started = DispatchTime.now().uptimeNanoseconds
    let observations = try await request.perform(on: imageURL)
    let latency = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
    let result: [String: Any] = [
        "coordinate_system": "normalized_top_left",
        "revision": "revision1",
        "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        "recognition_latency_ms": latency,
        "documents": observations.map { documentJSON($0.document) },
    ]
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fail("Apple Vision recognition failed: \(error)")
}
