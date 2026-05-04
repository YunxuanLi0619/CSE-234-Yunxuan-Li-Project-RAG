import AppKit
import CoreText
import Foundation

if CommandLine.arguments.count != 3 {
    fputs("usage: swift render_report_pdf.swift input.md output.pdf\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let source = try String(contentsOf: inputURL, encoding: .utf8)

let pageWidth: CGFloat = 612
let pageHeight: CGFloat = 792
let margin: CGFloat = 46
let contentRect = CGRect(x: margin, y: margin, width: pageWidth - 2 * margin, height: pageHeight - 2 * margin)

func paragraphStyle(spacingBefore: CGFloat = 0, spacingAfter: CGFloat = 4, lineSpacing: CGFloat = 1.4) -> NSMutableParagraphStyle {
    let style = NSMutableParagraphStyle()
    style.paragraphSpacingBefore = spacingBefore
    style.paragraphSpacing = spacingAfter
    style.lineSpacing = lineSpacing
    return style
}

func append(_ text: String, to out: NSMutableAttributedString, font: NSFont, spacingBefore: CGFloat = 0, spacingAfter: CGFloat = 4) {
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor.black,
        .paragraphStyle: paragraphStyle(spacingBefore: spacingBefore, spacingAfter: spacingAfter),
    ]
    out.append(NSAttributedString(string: text, attributes: attrs))
}

let rendered = NSMutableAttributedString()
for rawLine in source.split(separator: "\n", omittingEmptySubsequences: false) {
    let line = String(rawLine)
    if line.hasPrefix("# ") {
        append(String(line.dropFirst(2)) + "\n", to: rendered, font: .boldSystemFont(ofSize: 17), spacingAfter: 10)
    } else if line.hasPrefix("## ") {
        append(String(line.dropFirst(3)) + "\n", to: rendered, font: .boldSystemFont(ofSize: 13.5), spacingBefore: 6, spacingAfter: 5)
    } else if line.hasPrefix("|") {
        append(line + "\n", to: rendered, font: NSFont.monospacedSystemFont(ofSize: 7.5, weight: .regular), spacingAfter: 1.5)
    } else if line.hasPrefix("```") {
        continue
    } else if line.hasPrefix("- ") || line.hasPrefix("* ") {
        append(line + "\n", to: rendered, font: .systemFont(ofSize: 9.4), spacingAfter: 2.5)
    } else if line.trimmingCharacters(in: .whitespaces).isEmpty {
        append("\n", to: rendered, font: .systemFont(ofSize: 6), spacingAfter: 0)
    } else {
        append(line + "\n", to: rendered, font: .systemFont(ofSize: 9.4), spacingAfter: 4)
    }
}

let framesetter = CTFramesetterCreateWithAttributedString(rendered as CFAttributedString)
let data = NSMutableData()
var mediaBox = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)
guard let consumer = CGDataConsumer(data: data as CFMutableData),
      let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
    fputs("failed to create PDF context\n", stderr)
    exit(1)
}

var current = CFRange(location: 0, length: 0)
let totalLength = rendered.length
var pageNumber = 1

while current.location < totalLength {
    context.beginPDFPage(nil)
    context.saveGState()
    context.setFillColor(NSColor.white.cgColor)
    context.fill(CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight))
    context.textMatrix = .identity

    let path = CGMutablePath()
    path.addRect(contentRect)
    let frame = CTFramesetterCreateFrame(framesetter, current, path, nil)
    CTFrameDraw(frame, context)
    let visible = CTFrameGetVisibleStringRange(frame)
    current.location += max(visible.length, 1)
    context.restoreGState()

    let footer = "Team 40 Project Report - \(pageNumber)" as NSString
    let footerAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 8),
        .foregroundColor: NSColor.darkGray,
    ]
    footer.draw(at: CGPoint(x: margin, y: 22), withAttributes: footerAttrs)

    context.endPDFPage()
    pageNumber += 1
}

context.closePDF()
try data.write(to: outputURL, options: .atomic)
print("Wrote \(outputURL.path)")
