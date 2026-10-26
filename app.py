from pathlib import Path
from datetime import datetime
import csv
import io
import html

import joblib
import pandas as pd
import streamlit as st

# Optional PDF support
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PATHS AND MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

FEATURES = [
    "age",
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "paid",
    "activities",
    "higher",
    "traveltime",
    "goout",
]


# =========================================================
# SESSION STATE DEFAULTS
# =========================================================

DEFAULTS = {
    "student_result": None,
    "first_name": "",
    "surname": "",
    "student_id": "",
    "age": 17,
    "studytime_label": "Less than 2 hours",
    "failures_input": 0,
    "absences": 3,
    "schoolsup": "No",
    "paid": "No",
    "activities": "No",
    "higher": "Yes",
    "travel_label": "15–30 minutes",
    "goout_label": "Moderate",
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def reset_form():
    """Reset the form and clear the previous prediction."""
    for key, default_value in DEFAULTS.items():
        st.session_state[key] = default_value


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# INPUT MAPPINGS
# =========================================================

STUDYTIME_OPTIONS = {
    "Less than 2 hours": 1,
    "2–5 hours": 2,
    "5–10 hours": 3,
    "More than 10 hours": 4,
}

TRAVEL_OPTIONS = {
    "Less than 15 minutes": 1,
    "15–30 minutes": 2,
    "30 minutes–1 hour": 3,
    "More than 1 hour": 4,
}

GOOUT_OPTIONS = {
    "Very low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very high": 5,
}


# =========================================================
# DASHBOARD STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                #172b4d 0,
                #0b1220 40%,
                #080d18 100%
            );
        color: #e8eef8;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #f3f7ff;
    }

    p, label, .stMarkdown {
        color: #d7e1f0;
    }

    .hero {
        padding: 26px 30px;
        border: 1px solid #2b4264;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            rgba(24, 44, 75, 0.98),
            rgba(13, 23, 42, 0.98)
        );
        margin-bottom: 24px;
    }

    .hero-title {
        color: #f5f8ff;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #b5c7e2;
        font-size: 15px;
        margin-bottom: 0;
    }

    .paneltitle {
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .muted {
        color: #9eb0ca;
        font-size: 13px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #2b3d58 !important;
        background: rgba(13, 23, 40, 0.78);
        border-radius: 16px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #101c30;
        color: #f3f7ff;
        border: 1px solid #344968;
        border-radius: 9px;
    }

    [data-testid="stSelectbox"] > div > div {
        background: #101c30;
        color: #f3f7ff;
        border-radius: 9px;
    }

    div[data-baseweb="select"] > div {
        background-color: #101c30;
        border-color: #344968;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #385477;
        background: #18365c;
        color: #f4f8ff;
        font-weight: 650;
        min-height: 42px;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #75b8ff;
        background: #244a78;
        color: white;
    }

    .primary-result {
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            rgba(16, 75, 57, 0.95),
            rgba(12, 45, 40, 0.95)
        );
        border: 1px solid #26795b;
        margin: 12px 0 20px 0;
    }

    .result-label {
        color: #a9e9cf;
        font-size: 14px;
        font-weight: 600;
    }

    .result-grade {
        color: #70efb5;
        font-size: 50px;
        line-height: 1.2;
        font-weight: 850;
        margin: 6px 0;
    }

    .result-description {
        color: #c7f6e3;
        font-size: 14px;
    }

    .factor-card {
        padding: 15px;
        border-radius: 12px;
        background: rgba(73, 45, 120, 0.24);
        border: 1px solid #644c91;
        margin: 10px 0;
    }

    .factor-title {
        color: #d9c8ff;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .factor-text {
        color: #c3cce0;
        font-size: 13px;
    }

    .tip-card {
        padding: 15px;
        border-radius: 12px;
        background: rgba(103, 77, 19, 0.23);
        border: 1px solid #8b702c;
        margin: 10px 0;
    }

    .tip-title {
        color: #ffdc82;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .tip-text {
        color: #e6dcc1;
        font-size: 13px;
    }

    .student-details {
        padding: 15px;
        border: 1px solid #2b4264;
        border-radius: 12px;
        background: rgba(15, 27, 46, 0.85);
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #8092ad;
        font-size: 12px;
        padding-top: 22px;
    }

    hr {
        border-color: #2b3d58;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_text(value):
    """Escape text before inserting it into HTML."""
    return html.escape(str(value))


def get_grade_description(grade):
    """Return a simple interpretation of the predicted grade."""
    if grade >= 18:
        return "Excellent predicted performance"
    elif grade >= 15:
        return "Very good predicted performance"
    elif grade >= 12:
        return "Good predicted performance"
    elif grade >= 10:
        return "Satisfactory predicted performance"
    else:
        return "Additional study and support may be helpful"


def get_key_factors(inputs):
    """Create simple explanations based on the entered information."""
    factors = []

    studytime = inputs["studytime"]
    failures = inputs["failures"]
    absences = inputs["absences"]
    schoolsup = inputs["schoolsup"]
    higher = inputs["higher"]
    goout = inputs["goout"]

    if studytime >= 3:
        factors.append(
            (
                "Study time",
                "You selected a relatively high weekly study-time category. "
                "Consistent revision may support better academic performance.",
            )
        )
    else:
        factors.append(
            (
                "Study time",
                "A regular study routine and additional revision time "
                "may help reinforce your understanding.",
            )
        )

    if failures > 0:
        factors.append(
            (
                "Previous failures",
                f"You entered {failures} previous class failure(s). "
                "Reviewing difficult topics and seeking support may be useful.",
            )
        )
    else:
        factors.append(
            (
                "Previous failures",
                "You entered no previous class failures. "
                "Continue reviewing material regularly to maintain progress.",
            )
        )

    if absences > 10:
        factors.append(
            (
                "Absences",
                f"You entered {absences} absences. "
                "Attending classes consistently can help you keep up with lessons.",
            )
        )
    else:
        factors.append(
            (
                "Absences",
                f"You entered {absences} absences. "
                "Keeping attendance consistent can help support learning.",
            )
        )

    if schoolsup == "Yes":
        factors.append(
            (
                "School support",
                "You indicated that school support is available. "
                "Using that support can help clarify difficult topics.",
            )
        )

    if higher == "Yes":
        factors.append(
            (
                "Further education",
                "You indicated an interest in higher education. "
                "Setting clear academic goals may help you stay motivated.",
            )
        )

    if goout >= 4:
        factors.append(
            (
                "Social activities",
                "You selected a high level of going out. "
                "Balancing social activities with study time may help maintain a routine.",
            )
        )

    return factors[:4]


def get_study_tips(inputs, grade):
    """Generate practical study suggestions."""
    tips = []

    if inputs["studytime"] <= 2:
        tips.append(
            (
                "Build a study routine",
                "Try short, focused study sessions during the week "
                "and review difficult material regularly.",
            )
        )

    if inputs["failures"] > 0:
        tips.append(
            (
                "Review challenging topics",
                "Identify topics that caused difficulties previously. "
                "Practise similar exercises and ask for help when needed.",
            )
        )

    if inputs["absences"] > 10:
        tips.append(
            (
                "Keep up with lessons",
                "Review missed lessons and obtain notes or explanations "
                "for topics you did not attend.",
            )
        )

    if grade < 10:
        tips.append(
            (
                "Focus on the fundamentals",
                "Break topics into smaller parts, practise basic exercises, "
                "and check your understanding after each session.",
            )
        )

    if not tips:
        tips.append(
            (
                "Stay consistent",
                "Continue revising regularly, practise exercises, "
                "and track the topics you understand least.",
            )
        )

    tips.append(
        (
            "Plan your week",
            "Set a few realistic academic goals each week and make time "
            "for rest as well as studying.",
        )
    )

    return tips[:3]


# =========================================================
# CSV EXPORT
# =========================================================

def build_csv(result):
    """Build a CSV report that can be opened in spreadsheet software."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["Student Performance Predictor Report"])
    writer.writerow(["First name", result["first_name"]])
    writer.writerow(["Surname", result["surname"]])
    writer.writerow(["Student ID", result["student_id"]])
    writer.writerow(["Estimated grade", f'{result["grade"]:.2f} / 20'])
    writer.writerow(["Model used", result["model"]])
    writer.writerow(["Prediction date", result["date"]])

    writer.writerow([])
    writer.writerow(["Academic input", "Value"])

    for key, value in result["display_inputs"].items():
        writer.writerow([key, value])

    # UTF-8 BOM helps Excel display characters correctly.
    return "\ufeff" + buffer.getvalue()


# =========================================================
# PDF EXPORT
# =========================================================

def build_pdf(result):
    """Build a downloadable PDF report."""
    if not PDF_AVAILABLE:
        return None

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitleCustom",
            parent=styles["Title"],
            textColor=colors.HexColor("#18365c"),
            fontSize=20,
            leading=25,
            spaceAfter=14,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitleCustom",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#244a78"),
            fontSize=13,
            leading=17,
            spaceBefore=12,
            spaceAfter=7,
        )
    )

    story = []

    story.append(
        Paragraph(
            "Student Performance Predictor",
            styles["ReportTitleCustom"],
        )
    )

    story.append(
        Paragraph(
            "Student prediction report",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 14))

    student_data = [
        ["Student information", "Details"],
        ["First name", result["first_name"] or "Not provided"],
        ["Surname", result["surname"] or "Not provided"],
        ["Student ID", result["student_id"] or "Not provided"],
        ["Estimated grade", f'{result["grade"]:.2f} / 20'],
        ["Interpretation", get_grade_description(result["grade"])],
        ["Model used", result["model"]],
        ["Prediction date", result["date"]],
    ]

    student_table = Table(
        student_data,
        colWidths=[5 * cm, 10 * cm],
        repeatRows=1,
    )

    student_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#18365c"),
                ),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4dce8")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f8fc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(student_table)
    story.append(Spacer(1, 16))

    story.append(
        Paragraph("Academic inputs", styles["SectionTitleCustom"])
    )

    input_data = [["Input", "Value"]]

    for key, value in result["display_inputs"].items():
        input_data.append([str(key), str(value)])

    input_table = Table(
        input_data,
        colWidths=[7 * cm, 8 * cm],
        repeatRows=1,
    )

    input_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#244a78"),
                ),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4dce8")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(input_table)
    story.append(Spacer(1, 16))

    story.append(
        Paragraph("Study recommendations", styles["SectionTitleCustom"])
    )

    for title, description in get_study_tips(
        result["inputs"], result["grade"]
    ):
        story.append(
            Paragraph(
                f"<b>{safe_text(title)}</b>: {safe_text(description)}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 7))

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Disclaimer: This result is a machine-learning estimate based "
            "on the supplied information. It is not a guaranteed final grade "
            "or a professional academic assessment.",
            styles["Italic"],
        )
    )

    document.build(story)

    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎓 Student Performance Predictor</div>
        <p class="hero-subtitle">
            Estimate a student's final mathematics grade using a trained
            machine-learning model. Explore the result and practical
            recommendations in one dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL AVAILABILITY CHECK
# =========================================================

if model is None:
    st.error(
        "The trained model could not be found. "
        "Please check that the model exists at "
        "`models/best_model.joblib`."
    )
    st.stop()


# =========================================================
# MAIN DASHBOARD
# =========================================================

left, right = st.columns([1, 1], gap="large")


# =========================================================
# LEFT COLUMN — STUDENT INPUTS
# =========================================================

with left:
    with st.container(border=True):
        st.markdown(
            '<div class="paneltitle">👤 Student Information</div>'
            '<p class="muted">Enter the student details below.</p>',
            unsafe_allow_html=True,
        )

        st.text_input(
            "First name",
            key="first_name",
            placeholder="Enter first name",
        )

        st.text_input(
            "Surname",
            key="surname",
            placeholder="Enter surname",
        )

        st.text_input(
            "Student ID",
            key="student_id",
            placeholder="Enter student ID",
        )

    with st.container(border=True):
        st.markdown(
            '<div class="paneltitle">📚 Academic Information</div>'
            '<p class="muted">Enter the academic factors used for prediction.</p>',
            unsafe_allow_html=True,
        )

        st.number_input(
            "🎂 Student age",
            min_value=15,
            max_value=30,
            step=1,
            key="age",
        )

        st.selectbox(
            "📖 Weekly study time",
            options=list(STUDYTIME_OPTIONS.keys()),
            key="studytime_label",
            help="Select the approximate weekly study-time category.",
        )

        st.number_input(
            "⚠️ Previous class failures",
            min_value=0,
            max_value=1000,
            step=1,
            format="%d",
            key="failures_input",
            help="Enter the number of previous class failures.",
        )

        st.number_input(
            "🗓️ Number of absences",
            min_value=0,
            max_value=1000,
            step=1,
            key="absences",
        )

        st.selectbox(
            "🧑‍🏫 Extra educational support",
            options=["No", "Yes"],
            key="schoolsup",
        )

        st.selectbox(
            "💳 Extra paid classes",
            options=["No", "Yes"],
            key="paid",
        )

        st.selectbox(
            "⚽ Extracurricular activities",
            options=["No", "Yes"],
            key="activities",
        )

        st.selectbox(
            "🎓 Interested in higher education",
            options=["No", "Yes"],
            key="higher",
        )

        st.selectbox(
            "🚌 Travel time to school",
            options=list(TRAVEL_OPTIONS.keys()),
            key="travel_label",
        )

        st.selectbox(
            "👥 Going out with friends",
            options=list(GOOUT_OPTIONS.keys()),
            key="goout_label",
        )

        st.markdown("---")

        predict_col, reset_col = st.columns(2)

        predict_clicked = predict_col.button(
            "🔮 Predict Grade",
            use_container_width=True,
            type="primary",
        )

        reset_col.button(
            "🔄 Reset Form",
            use_container_width=True,
            on_click=reset_form,
        )


# =========================================================
# PREDICTION LOGIC
# =========================================================

if predict_clicked:
    input_values = {
        "age": int(st.session_state["age"]),
        "studytime": STUDYTIME_OPTIONS[
            st.session_state["studytime_label"]
        ],
        "failures": int(st.session_state["failures_input"]),
        "absences": int(st.session_state["absences"]),
        "schoolsup": st.session_state["schoolsup"],
        "paid": st.session_state["paid"],
        "activities": st.session_state["activities"],
        "higher": st.session_state["higher"],
        "traveltime": TRAVEL_OPTIONS[
            st.session_state["travel_label"]
        ],
        "goout": GOOUT_OPTIONS[
            st.session_state["goout_label"]
        ],
    }

    # The order of these columns must match the order used during training.
    prediction_df = pd.DataFrame(
        [[input_values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    try:
        prediction = model.predict(prediction_df)[0]

        # The model predicts G3 on the original 0–20 grade scale.
        predicted_grade = max(0.0, min(20.0, float(prediction)))

        display_inputs = {
            "Age": input_values["age"],
            "Weekly study time": st.session_state["studytime_label"],
            "Previous class failures": input_values["failures"],
            "Absences": input_values["absences"],
            "Extra educational support": input_values["schoolsup"],
            "Extra paid classes": input_values["paid"],
            "Extracurricular activities": input_values["activities"],
            "Interested in higher education": input_values["higher"],
            "Travel time to school": st.session_state["travel_label"],
            "Going out with friends": st.session_state["goout_label"],
        }

        st.session_state["student_result"] = {
            "first_name": st.session_state["first_name"].strip(),
            "surname": st.session_state["surname"].strip(),
            "student_id": st.session_state["student_id"].strip(),
            "grade": predicted_grade,
            "model": type(model).__name__,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "inputs": input_values,
            "display_inputs": display_inputs,
        }

    except Exception as error:
        st.error(
            "Prediction failed. Please check that the saved model "
            "matches the features and input format used in this app."
        )
        st.exception(error)


# =========================================================
# RIGHT COLUMN — RESULTS
# =========================================================

with right:
    result = st.session_state["student_result"]

    if result is None:
        with st.container(border=True):
            st.markdown(
                '<div class="paneltitle">📊 Prediction Results</div>'
                '<p class="muted">'
                'Your results will appear here after you click '
                '<b>Predict Grade</b>.'
                '</p>',
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="primary-result">
                    <div class="result-label">Estimated final grade</div>
                    <div class="result-grade">-- / 20</div>
                    <div class="result-description">
                        Complete the student form to generate a prediction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "The prediction is an estimate from a machine-learning model, "
                "not a guaranteed final grade."
            )

    else:
        # -------------------------------------------------
        # STUDENT DETAILS
        # -------------------------------------------------

        with st.container(border=True):
            st.markdown(
                '<div class="paneltitle">📊 Prediction Results</div>',
                unsafe_allow_html=True,
            )

            student_name = (
                f'{result["first_name"]} {result["surname"]}'.strip()
            )

            if not student_name:
                student_name = "Student"

            st.markdown(
                f"""
                <div class="student-details">
                    <b>Student:</b> {safe_text(student_name)}<br>
                    <b>Student ID:</b>
                    {safe_text(result["student_id"] or "Not provided")}<br>
                    <b>Model:</b> {safe_text(result["model"])}<br>
                    <b>Prediction date:</b> {safe_text(result["date"])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            grade = result["grade"]

            st.markdown(
                f"""
                <div class="primary-result">
                    <div class="result-label">Estimated final grade</div>
                    <div class="result-grade">{grade:.2f} / 20</div>
                    <div class="result-description">
                        {safe_text(get_grade_description(grade))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Grade scale**")

            st.progress(
                max(0.0, min(1.0, grade / 20.0)),
                text=f"{grade:.2f} out of 20",
            )

            st.caption(
                "This is a model estimate and should not be interpreted "
                "as a guaranteed academic result."
            )

        # -------------------------------------------------
        # KEY FACTORS
        # -------------------------------------------------

        with st.container(border=True):
            st.markdown(
                '<div class="paneltitle" style="color:#d9c8ff">'
                '🔍 Key Factors</div>'
                '<p class="muted">'
                'Factors to consider based on the information you entered.'
                '</p>',
                unsafe_allow_html=True,
            )

            for factor_title, factor_text in get_key_factors(
                result["inputs"]
            ):
                st.markdown(
                    f"""
                    <div class="factor-card">
                        <div class="factor-title">
                            {safe_text(factor_title)}
                        </div>
                        <div class="factor-text">
                            {safe_text(factor_text)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------
        # STUDY TIPS
        # -------------------------------------------------

        with st.container(border=True):
            st.markdown(
                '<div class="paneltitle" style="color:#ffdc82">'
                '💡 Study Tips</div>'
                '<p class="muted">'
                'Practical suggestions based on the supplied information.'
                '</p>',
                unsafe_allow_html=True,
            )

            for tip_title, tip_text in get_study_tips(
                result["inputs"], result["grade"]
            ):
                st.markdown(
                    f"""
                    <div class="tip-card">
                        <div class="tip-title">
                            {safe_text(tip_title)}
                        </div>
                        <div class="tip-text">
                            {safe_text(tip_text)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------
        # EXPORT RESULTS — CSV + PDF
        # -------------------------------------------------

        with st.container(border=True):
            st.markdown(
                '<div class="paneltitle" style="color:#b8e8ff">'
                '📄 Export Results</div>'
                '<p class="muted">'
                'Download the student results for your records.'
                '</p>',
                unsafe_allow_html=True,
            )

            x1, x2 = st.columns(2)

            safe_id = "".join(
                char
                for char in result["student_id"]
                if char.isalnum() or char in ("-", "_")
            ) or "student"

            x1.download_button(
                "⬇ Download as CSV",
                data=build_csv(result),
                file_name=f"student_result_{safe_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if PDF_AVAILABLE:
                x2.download_button(
                    "⬇ Download as PDF",
                    data=build_pdf(result),
                    file_name=f"student_result_{safe_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                x2.warning("Install ReportLab to enable PDF export.")
                st.code(
                    r".\.venv\Scripts\python.exe -m pip install reportlab",
                    language="powershell",
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Student Performance Predictor · Machine Learning Project<br>
        Predictions are estimates and do not guarantee a student's final grade.
    </div>
    """,
    unsafe_allow_html=True,
)