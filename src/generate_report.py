"""Generate comprehensive PDF report for UnblurML project."""

import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT


PAGE_W, PAGE_H = letter
MARGIN = 54
CONTENT_W = PAGE_W - 2 * MARGIN

# Colors
NAVY = colors.HexColor("#0A1628")
DARK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#3B82F6")
GREEN = colors.HexColor("#22C55E")
RED = colors.HexColor("#EF4444")
ORANGE = colors.HexColor("#F59E0B")
GRAY = colors.HexColor("#6B7280")
LIGHT_BG = colors.HexColor("#F8FAFC")
WHITE = colors.white


def draw_header_footer(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 44, PAGE_W, 44, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(MARGIN, PAGE_H - 30, "UnblurML")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 28, "BIP-39 Seed Phrase Deblurring Model")

    # Footer
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 36, PAGE_W - MARGIN, 36)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN, 24, "UnblurML — Confidential")
    canvas.drawRightString(PAGE_W - MARGIN, 24, f"Page {doc.page}")
    canvas.restoreState()


def draw_cover(canvas, doc):
    canvas.saveState()
    # Full-page dark background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H * 0.52, PAGE_W, 4, fill=1, stroke=0)

    # Title
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 36)
    canvas.drawString(MARGIN, PAGE_H * 0.65, "UnblurML")

    canvas.setFont("Helvetica", 16)
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.drawString(MARGIN, PAGE_H * 0.60, "BIP-39 Seed Phrase Deblurring Model")

    # Subtitle
    canvas.setFont("Helvetica", 12)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(MARGIN, PAGE_H * 0.45, "Training Report & Benchmark Analysis")
    canvas.drawString(MARGIN, PAGE_H * 0.42, "March 2026")

    # Key stats
    y = PAGE_H * 0.30
    stats = [
        ("87.4%", "Best Val Top-1 Accuracy"),
        ("96.4%", "Best Val Top-5 Accuracy"),
        ("14 min", "Total Training Time"),
        ("2,048", "BIP-39 Word Classes"),
    ]
    canvas.setFont("Helvetica-Bold", 20)
    for val, label in stats:
        canvas.setFillColor(ACCENT)
        canvas.drawString(MARGIN, y, val)
        canvas.setFillColor(colors.HexColor("#94A3B8"))
        canvas.setFont("Helvetica", 10)
        canvas.drawString(MARGIN + 100, y + 2, label)
        canvas.setFont("Helvetica-Bold", 20)
        y -= 32

    canvas.restoreState()


def build_report():
    output_path = "reports/UnblurML_Report.pdf"

    doc = BaseDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=60,
        bottomMargin=54,
    )

    frame_cover = Frame(0, 0, PAGE_W, PAGE_H, id="cover")
    frame_content = Frame(
        MARGIN, 54, CONTENT_W, PAGE_H - 60 - 54,
        id="content",
    )

    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[frame_cover], onPage=draw_cover),
        PageTemplate(id="Content", frames=[frame_content], onPage=draw_header_footer),
    ])

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        "SectionTitle", parent=styles["Heading1"],
        fontSize=18, textColor=NAVY, spaceBefore=24, spaceAfter=12,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "SubTitle", parent=styles["Heading2"],
        fontSize=13, textColor=DARK, spaceBefore=16, spaceAfter=8,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=DARK, leading=15, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=GRAY, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=16, italic=True,
    ))
    styles.add(ParagraphStyle(
        "Metric", parent=styles["Normal"],
        fontSize=11, textColor=NAVY, fontName="Helvetica-Bold",
    ))

    from reportlab.platypus.doctemplate import NextPageTemplate
    story = []

    # --- Cover Page ---
    story.append(NextPageTemplate("Content"))
    story.append(PageBreak())

    # --- Executive Summary ---
    story.append(Paragraph("Executive Summary", styles["SectionTitle"]))
    story.append(Paragraph(
        "UnblurML is a deep learning model that recovers Gaussian-blurred BIP-39 seed phrase words. "
        "Given a blurred image of a single word, the model classifies it as one of the 2,048 words in the "
        "BIP-39 English wordlist. This constrained vocabulary makes the problem tractable as classification "
        "rather than reconstruction — we don't need to reconstruct pixel-perfect text, just identify which word it is.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "The model achieves <b>99.5% top-1 accuracy at light blur</b> (sigma 1), <b>91% at mild blur</b> (sigma 3), "
        "and <b>83% top-5 at medium blur</b> (sigma 5). Training completed in <b>14 minutes</b> on an Apple M4 Max "
        "with 64GB unified memory using PyTorch MPS backend.",
        styles["Body"],
    ))
    story.append(Spacer(1, 16))

    # --- Architecture ---
    story.append(Paragraph("Model Architecture", styles["SectionTitle"]))

    arch_data = [
        ["Component", "Choice", "Rationale"],
        ["Backbone", "ResNet-18 (pretrained ImageNet)", "Fast convergence, small footprint (12M params)"],
        ["Classifier", "2048-class FC head", "One output per BIP-39 word"],
        ["Input Size", "64 x 192 px (RGB)", "Matches typical word aspect ratio"],
        ["Normalization", "Mean/Std = 0.5", "Grayscale-appropriate (not ImageNet stats)"],
        ["Loss", "CrossEntropy + label smoothing 0.1", "Prevents overconfident predictions"],
        ["Optimizer", "AdamW + OneCycleLR", "Fast convergence with LR warmup"],
        ["Augmentation", "JPEG, Affine, CoarseDropout, Brightness", "Simulates real-world degradation"],
    ]
    wrapped = [[Paragraph(str(c), styles["Body"]) for c in row] for row in arch_data]
    wrapped[0] = [Paragraph(str(c), styles["Metric"]) for c in arch_data[0]]

    col_ws = [CONTENT_W * 0.22, CONTENT_W * 0.35, CONTENT_W * 0.43]
    t = Table(wrapped, colWidths=col_ws, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Curriculum learning
    story.append(Paragraph("Curriculum Learning Strategy", styles["SubTitle"]))
    story.append(Paragraph(
        "Training uses curriculum learning — the model starts with easy examples and gradually increases difficulty. "
        "This prevents the optimizer from getting stuck at the random baseline (loss = log(2048) = 7.62).",
        styles["Body"],
    ))

    curr_data = [
        ["Phase", "Epochs", "Blur Sigma Range", "Purpose"],
        ["1 — Warmup", "1", "0.5 – 3.0", "Frozen backbone, train head only"],
        ["2 — Easy", "2–5", "0.5 – 3.0", "Learn word shapes with minimal blur"],
        ["3 — Medium", "6–10", "0.5 – 5.0", "Generalize to medium blur"],
        ["4 — Hard", "11–15", "0.5 – 7.0", "Push to maximum recoverable blur"],
    ]
    wrapped = [[Paragraph(str(c), styles["Body"]) for c in row] for row in curr_data]
    wrapped[0] = [Paragraph(str(c), styles["Metric"]) for c in curr_data[0]]
    col_ws2 = [CONTENT_W * 0.20, CONTENT_W * 0.12, CONTENT_W * 0.25, CONTENT_W * 0.43]
    t2 = Table(wrapped, colWidths=col_ws2, repeatRows=1)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)
    story.append(PageBreak())

    # --- Training Results ---
    story.append(Paragraph("Training Results", styles["SectionTitle"]))
    story.append(Paragraph(
        "The model was trained for 15 epochs in 14 minutes. Best checkpoint was saved at epoch 10 "
        "with <b>87.4% top-1</b> and <b>96.4% top-5</b> validation accuracy (sigma range 0.5–5.0).",
        styles["Body"],
    ))
    story.append(Image("reports/examples/training_curves.png", width=CONTENT_W, height=CONTENT_W * 0.36))
    story.append(Paragraph("Training accuracy and loss progression across 15 epochs with curriculum learning phases.", styles["Caption"]))

    # Training history table
    with open("models/resnet18_history.json") as f:
        history = json.load(f)

    hist_data = [["Epoch", "Sigma Range", "Train Acc", "Val Top-1", "Val Top-5", "Time"]]
    for h in history:
        hist_data.append([
            str(h["epoch"]),
            f"{h['sigma_range'][0]}–{h['sigma_range'][1]}",
            f"{h['train_acc']:.1f}%",
            f"{h['val_top1']:.1f}%",
            f"{h['val_top5']:.1f}%",
            f"{h['epoch_time']:.0f}s",
        ])
    wrapped = [[Paragraph(str(c), styles["Body"]) for c in row] for row in hist_data]
    wrapped[0] = [Paragraph(str(c), styles["Metric"]) for c in hist_data[0]]
    cw = [CONTENT_W * w for w in [0.10, 0.18, 0.18, 0.18, 0.18, 0.18]]
    t3 = Table(wrapped, colWidths=cw, repeatRows=1)
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t3)
    story.append(PageBreak())

    # --- Blur Level Performance ---
    story.append(Paragraph("Accuracy by Blur Level", styles["SectionTitle"]))
    story.append(Image("reports/sigma_accuracy.png", width=CONTENT_W, height=CONTENT_W * 0.55))
    story.append(Paragraph("Model accuracy degrades gracefully from sigma 1–5, then drops sharply as information is physically destroyed.", styles["Caption"]))

    sigma_data = [["Sigma", "Top-1", "Top-5", "Assessment"]]
    assessments = {
        1: ("99.5%", "100.0%", "Near-perfect — text clearly readable"),
        3: ("91.0%", "98.0%", "Excellent — letters blurry but distinguishable"),
        5: ("62.5%", "83.0%", "Usable — word shapes still visible"),
        8: ("5.5%", "13.0%", "Poor — kernel exceeds character features"),
        12: ("0.5%", "0.5%", "Impossible — image is featureless"),
    }
    for sigma, (t1, t5, desc) in assessments.items():
        sigma_data.append([str(sigma), t1, t5, desc])
    wrapped = [[Paragraph(str(c), styles["Body"]) for c in row] for row in sigma_data]
    wrapped[0] = [Paragraph(str(c), styles["Metric"]) for c in sigma_data[0]]
    cw4 = [CONTENT_W * w for w in [0.10, 0.12, 0.12, 0.66]]
    t4 = Table(wrapped, colWidths=cw4, repeatRows=1)
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#DCFCE7")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#DCFCE7")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#FEF9C3")),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#FEE2E2")),
        ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#FEE2E2")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t4)
    story.append(PageBreak())

    # --- Visual Examples ---
    story.append(Paragraph("Visual Examples", styles["SectionTitle"]))

    story.append(Paragraph("Successful Predictions", styles["SubTitle"]))
    story.append(Image("reports/examples/success_showcase.png", width=CONTENT_W, height=CONTENT_W * 0.50))
    story.append(Paragraph("Original words (left) and their blurred versions with model predictions (right). Green = correct, Red = incorrect.", styles["Caption"]))

    story.append(Paragraph("Prediction Confidence at Different Blur Levels", styles["SubTitle"]))
    story.append(Image("reports/examples/confidence_bars.png", width=CONTENT_W, height=CONTENT_W * 0.39))
    story.append(Paragraph("Top-5 prediction confidence bars showing how certainty decreases with higher blur sigma.", styles["Caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Blur Level Grid", styles["SubTitle"]))
    story.append(Image("reports/blur_levels_grid.png", width=CONTENT_W, height=CONTENT_W * 0.67))
    story.append(Paragraph("Each row shows one BIP-39 word at increasing Gaussian blur sigma (2, 5, 10, 15, 20, 25). Green titles = correct prediction.", styles["Caption"]))
    story.append(PageBreak())

    # --- Edge Cases ---
    story.append(Paragraph("Edge Case Analysis", styles["SectionTitle"]))
    story.append(Paragraph(
        "Real-world blurred images suffer from additional degradation: JPEG compression, partial cropping, "
        "noise from screenshots, and font variations. The benchmark evaluates each edge case independently.",
        styles["Body"],
    ))

    story.append(Image("reports/edge_cases_accuracy.png", width=CONTENT_W, height=CONTENT_W * 0.45))
    story.append(Paragraph("Top-5 accuracy across edge case scenarios. All tested with sigma 3–15 base blur.", styles["Caption"]))

    story.append(Image("reports/examples/failure_analysis.png", width=CONTENT_W, height=CONTENT_W * 0.56))
    story.append(Paragraph("Failure analysis at increasing difficulty. The model breaks down when blur sigma exceeds 6 or when multiple degradations combine.", styles["Caption"]))
    story.append(PageBreak())

    # --- Recommendations ---
    story.append(Paragraph("Recommendations & Next Steps", styles["SectionTitle"]))

    recs = [
        ("<b>Train longer with edge case augmentation</b> — Include heavier JPEG compression, cropping, and noise in the training loop. Current model was only trained on clean blur; 1–2 hours of training with these augmentations should bring edge case accuracy to 50%+.",
        ),
        ("<b>Increase input resolution</b> — Moving from 64x192 to 128x384 would let the model handle higher blur sigma, since the kernel size relative to image size determines the information limit.",
        ),
        ("<b>Try ConvNeXt V2 Tiny</b> — With more training time, a larger backbone will extract finer-grained features and improve accuracy across all blur levels.",
        ),
        ("<b>Ensemble multiple models</b> — Running ResNet-18 + ConvNeXt and taking majority vote would improve robustness on ambiguous cases.",
        ),
        ("<b>Add CLAHE preprocessing</b> — Contrast Limited Adaptive Histogram Equalization can enhance subtle gradients in blurred images before classification.",
        ),
        ("<b>Test on real screenshots</b> — Run the inference pipeline on actual blurred seed phrase screenshots to validate real-world performance.",
        ),
    ]
    for i, (rec,) in enumerate(recs):
        story.append(Paragraph(f"{i+1}. {rec}", styles["Body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 24))
    story.append(Paragraph("Technical Environment", styles["SubTitle"]))
    env_data = [
        ["Hardware", "Apple M4 Max, 64GB Unified Memory"],
        ["Framework", "PyTorch 2.10 + MPS Backend"],
        ["Model Library", "timm 1.0.25 (Hugging Face)"],
        ["Augmentation", "Albumentations 2.0"],
        ["Training Time", "14 minutes (15 epochs)"],
        ["Model Size", "12.2M parameters"],
    ]
    wrapped = [[Paragraph(str(c), styles["Body"]) for c in row] for row in env_data]
    cw5 = [CONTENT_W * 0.30, CONTENT_W * 0.70]
    t5 = Table(wrapped, colWidths=cw5)
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(t5)

    doc.build(story)
    print(f"Report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    build_report()
