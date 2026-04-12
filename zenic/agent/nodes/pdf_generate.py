"""Generate a formatted PDF from plan_data using FPDF2."""
import os
from fpdf import FPDF
from zenic.agent.state import ZenicState


def run(state: ZenicState) -> dict:
    intent = state.get("intent")
    plan_data = state.get("plan_data", {})
    output_path = f"/tmp/zenic_{intent}_plan.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    title = "Workout Plan" if intent == "workout_plan" else \
            "Meal Plan" if intent == "meal_plan" else "Weekly Summary"
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Zenic — {title}", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)

    # Render plan_data as formatted text — replace with proper templates per intent
    for key, value in plan_data.items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, str(key).replace("_", " ").title(), ln=True)
        pdf.set_font("Helvetica", size=10)
        if isinstance(value, list):
            for item in value:
                pdf.multi_cell(0, 6, f"  {item}")
        else:
            pdf.multi_cell(0, 6, f"  {value}")
        pdf.ln(2)

    pdf.output(output_path)
    return {
        "messages": [{
            "role": "assistant",
            "content": f"Your {title.lower()} is ready. Download it from the sidebar."
        }],
        "tool_results": {
            **(state.get("tool_results") or {}),
            "pdf_path": output_path,
        }
    }
