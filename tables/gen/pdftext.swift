// Converts a PDF to text with PDFKit: one "@@PAGE n" line, then the page text,
// pages separated by form feeds. Usage: pdftext in.pdf out.txt
import Foundation
import PDFKit

let args = CommandLine.arguments
guard args.count == 3, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])) else {
    FileHandle.standardError.write("usage: pdftext in.pdf out.txt\n".data(using: .utf8)!)
    exit(2)
}
var out = ""
for i in 0..<doc.pageCount {
    out += "\u{0C}@@PAGE \(i + 1)\n"
    out += (doc.page(at: i)?.string ?? "") + "\n"
}
do {
    try out.write(toFile: args[2], atomically: true, encoding: .utf8)
} catch {
    FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
    exit(1)
}
