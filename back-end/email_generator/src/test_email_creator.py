from fpdf import FPDF
import os

def create_sample_pdf():
    # Create samples directory if it doesn't exist
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add sample email content
    content = [
        "Dear Team,",
        "",
        "I hope this email finds you well. I wanted to touch base regarding our upcoming project timeline.",
        "",
        "As discussed in our last meeting, we need to finalize the deliverables by next week.",
        "Please review the attached documents and provide your feedback at your earliest convenience.",
        "",
        "Looking forward to your response.",
        "",
        "Best regards,",
        "John Smith"
    ]
    
    # Write content to PDF
    for line in content:
        pdf.cell(200, 10, txt=line, ln=True, align='L')
    
    # Save the PDF
    output_path = os.path.join(samples_dir, 'sample_email.pdf')
    pdf.output(output_path)
    print(f"Created sample PDF at: {output_path}")

if __name__ == "__main__":
    create_sample_pdf()