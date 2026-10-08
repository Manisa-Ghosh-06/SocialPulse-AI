from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form,
    Header
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from pathlib import Path

import pandas as pd

import io
import uuid
import json

import joblib

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.units import mm

from preprocessing import preprocess_text
from analysis import run_analysis


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="SocialPulse AI",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    vectorizer = joblib.load(
        BASE_DIR / "bow_Vectorizer.pkl"
    )

    sentiment_model = joblib.load(
        BASE_DIR / "bow_sentiment_model.pkl"
    )

    MODEL_LOADED = True

    print("✓ Sentiment model loaded")

except Exception as e:
    vectorizer = None
    sentiment_model = None
    MODEL_LOADED = False
    print("✗ Model loading failed:", e)


# REPLACED GLOBAL LIST WITH A DICTIONARY TO STORE SESSIONS
user_reports = {}

ALLOWED_FIELDS = {
    "TEXT": "Post Text",
    "USER_ID": "User ID",
    "AGE": "Age",
    "GENDER": "Gender",
    "LOCATION": "Location",
    "ENTITY": "Entity / Topic",
    "MENTIONED_USER": "Mentioned User",
    "FOLLOWERS": "Followers"
}


@app.get("/")
async def home():
    index_file = BASE_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(index_file)


@app.get("/socialpulse-logo.png")
async def logo():
    logo_file = BASE_DIR / "socialpulse-logo.png"

    if not logo_file.exists():
        raise HTTPException(
            status_code=404,
            detail="socialpulse-logo.png not found."
        )

    return FileResponse(logo_file)


@app.get("/api/model-status")
async def model_status():

    classes = []

    if sentiment_model is not None:

        try:

            classes = [
                str(x)
                for x in sentiment_model.classes_
            ]

        except Exception:

            classes = []

    return {
        "model_loaded": MODEL_LOADED,
        "model": "bow_sentiment_model.pkl",
        "vectorizer": "bow_Vectorizer.pkl",
        "classes": classes
    }


@app.post("/api/inspect")
async def inspect_csv(
    file: UploadFile = File(...)
):

    try:

        content = await file.read()

        df = pd.read_csv(
            io.BytesIO(content)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {e}"
        )

    columns = []

    for column in df.columns:

        column_name = str(column).strip()

        columns.append({
            "name": column_name
        })

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": columns,
        "mapping_options": [
            {
                "value": key,
                "label": label
            }
            for key, label
            in ALLOWED_FIELDS.items()
        ] + [
            {
                "value": "OTHER",
                "label": "Other"
            }
        ]
    }


def validate_mapping(
    mapping,
    dataframe
):

    if not isinstance(
        mapping,
        dict
    ):

        raise ValueError(
            "Invalid mapping."
        )

    cleaned_mapping = {}

    for field in ALLOWED_FIELDS:

        column = mapping.get(field)

        if column is None:
            continue

        if column == "":
            continue

        if column not in dataframe.columns:

            raise ValueError(
                f"Column '{column}' does not exist."
            )

        cleaned_mapping[field] = column

    if "TEXT" not in cleaned_mapping:

        raise ValueError(
            "Please map one column as Post Text."
        )

    used_columns = {}

    for field, column in cleaned_mapping.items():

        if column in used_columns:

            previous_field = used_columns[column]

            raise ValueError(
                f"Column '{column}' cannot be "
                f"mapped to both "
                f"'{previous_field}' and "
                f"'{field}'."
            )

        used_columns[column] = field

    return cleaned_mapping


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    mapping_json: str = Form(...),
    session_id: str = Header(None) # ADDED SESSION ID
):
    if not session_id:
        raise HTTPException(
            status_code=400, 
            detail="Session ID is missing in headers."
        )

    if not MODEL_LOADED:
        raise HTTPException(
            status_code=500,
            detail="Sentiment model is not loaded."
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {e}"
        )

    try:
        raw_mapping = json.loads(mapping_json)
        mapping = validate_mapping(raw_mapping, df)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    try:
        analytics = run_analysis(
            df=df,
            mapping=mapping,
            vectorizer=vectorizer,
            sentiment_model=sentiment_model,
            preprocess_text=preprocess_text
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {e}"
        )

    report_id = str(uuid.uuid4())

    report = {
        "id": report_id,
        "filename": file.filename,
        "rows": len(df),
        "mapping": mapping,
        "analytics": analytics
    }

    # ISOLATE REPORT SAVING BY SESSION ID
    if session_id not in user_reports:
        user_reports[session_id] = []
        
    user_reports[session_id].append(report)

    return report


@app.get("/api/reports")
async def get_reports(session_id: str = Header(None)): # ADDED SESSION ID

    # RETURN EMPTY LIST IF NO SESSION DATA EXISTS FOR THIS USER
    if not session_id or session_id not in user_reports:
        return []

    return [
        {
            "id": report["id"],
            "filename": report["filename"],
            "rows": report["rows"]
        }
        for report in user_reports[session_id]
    ]


@app.get("/api/reports/{report_id}")
async def get_report(
    report_id: str,
    session_id: str = Header(None) # ADDED SESSION ID
):
    
    if not session_id or session_id not in user_reports:
        raise HTTPException(
            status_code=404,
            detail="Report not found."
        )

    # SEARCH ONLY IN THE USER'S OWN REPORTS
    for report in user_reports[session_id]:

        if report["id"] == report_id:
            return report

    raise HTTPException(
        status_code=404,
        detail="Report not found."
    )


def pdf_paragraph(
    text,
    style
):
    return Paragraph(
        str(text),
        style
    )


def pdf_table(
    data,
    col_widths=None
):

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123047")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.HexColor("#22d3ee")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TEXTCOLOR",
                (0, 1),
                (-1, -1),
                colors.HexColor("#1f2937")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#b7c9d3")
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#f7fbfd")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    return table


def build_pdf_report(report):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SocialPulseTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0e7490"),
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "SocialPulseSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=18
    )

    section_style = ParagraphStyle(
        "SocialPulseSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0e7490"),
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "SocialPulseBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5
    )

    small_style = ParagraphStyle(
        "SocialPulseSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=3
    )

    story = []

    analytics = report.get(
        "analytics",
        {}
    )

    overview = analytics.get(
        "overview"
    )

    sentiment = analytics.get(
        "sentiment"
    )

    demographics = analytics.get(
        "demographics"
    )

    topics = analytics.get(
        "topics"
    )

    influence = analytics.get(
        "influence"
    )

    generated_at = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )

    story.append(
        Paragraph(
            "SocialPulse AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Social Media Analytics Report",
            subtitle_style
        )
    )

    metadata = [
        [
            "Dataset",
            str(
                report.get(
                    "filename",
                    "Uploaded Dataset"
                )
            )
        ],
        [
            "Rows Analyzed",
            str(
                report.get(
                    "rows",
                    0
                )
            )
        ],
        [
            "Generated",
            generated_at
        ]
    ]

    metadata_table = Table(
        metadata,
        colWidths=[
            38 * mm,
            132 * mm
        ]
    )

    metadata_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#e0f7fa")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#334155")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cbd5e1")
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        metadata_table
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    # OVERVIEW
    if overview is not None:

        story.append(
            Paragraph(
                "1. Overview",
                section_style
            )
        )

        overview_data = [
            ["Metric", "Value"],
            [
                "Total Posts",
                str(
                    overview.get(
                        "total_posts",
                        0
                    )
                )
            ],
            [
                "Total Users",
                str(
                    overview.get(
                        "total_users"
                    )
                    if overview.get(
                        "total_users"
                    ) is not None
                    else "Not mapped"
                )
            ],
            [
                "Mapped Fields",
                str(
                    len(
                        report.get(
                            "mapping",
                            {}
                        )
                    )
                )
            ]
        ]

        story.append(
            pdf_table(
                overview_data,
                [
                    65 * mm,
                    105 * mm
                ]
            )
        )

    # SENTIMENT
    if sentiment is not None:

        story.append(
            Paragraph(
                "2. Sentiment Analysis",
                section_style
            )
        )

        sentiment_data = [
            ["Sentiment", "Count", "Percentage"],
            [
                "Positive",
                str(
                    sentiment.get(
                        "positive",
                        0
                    )
                ),
                f"{sentiment.get('positive_percentage', 0)}%"
            ],
            [
                "Negative",
                str(
                    sentiment.get(
                        "negative",
                        0
                    )
                ),
                f"{sentiment.get('negative_percentage', 0)}%"
            ],
            [
                "Neutral",
                str(
                    sentiment.get(
                        "neutral",
                        0
                    )
                ),
                f"{sentiment.get('neutral_percentage', 0)}%"
            ],
            [
                "Total",
                str(
                    sentiment.get(
                        "total",
                        0
                    )
                ),
                "100%"
            ]
        ]

        story.append(
            pdf_table(
                sentiment_data,
                [
                    65 * mm,
                    52 * mm,
                    53 * mm
                ]
            )
        )

    # DEMOGRAPHICS
    if demographics is not None:

        story.append(
            Paragraph(
                "3. Demographics",
                section_style
            )
        )

        if demographics.get("age"):

            story.append(
                Paragraph(
                    "Age Distribution",
                    body_style
                )
            )

            age_data = [
                ["Age Group", "Count"]
            ]

            for item in demographics["age"]:

                age_data.append([
                    str(
                        item.get(
                            "name",
                            ""
                        )
                    ),
                    str(
                        item.get(
                            "count",
                            0
                        )
                    )
                ])

            story.append(
                pdf_table(
                    age_data,
                    [
                        100 * mm,
                        70 * mm
                    ]
                )
            )

            story.append(
                Spacer(
                    1,
                    6
                )
            )

        if demographics.get("gender"):

            story.append(
                Paragraph(
                    "Gender Distribution",
                    body_style
                )
            )

            gender_data = [
                ["Gender", "Count"]
            ]

            for item in demographics["gender"]:

                gender_data.append([
                    str(
                        item.get(
                            "name",
                            ""
                        )
                    ),
                    str(
                        item.get(
                            "count",
                            0
                        )
                    )
                ])

            story.append(
                pdf_table(
                    gender_data,
                    [
                        100 * mm,
                        70 * mm
                    ]
                )
            )

            story.append(
                Spacer(
                    1,
                    6
                )
            )

        if demographics.get("location"):

            story.append(
                Paragraph(
                    "Top Locations",
                    body_style
                )
            )

            location_data = [
                ["Location", "Count"]
            ]

            for item in demographics["location"]:

                location_data.append([
                    str(
                        item.get(
                            "name",
                            ""
                        )
                    ),
                    str(
                        item.get(
                            "count",
                            0
                        )
                    )
                ])

            story.append(
                pdf_table(
                    location_data,
                    [
                        100 * mm,
                        70 * mm
                    ]
                )
            )

    # TOPICS
    if topics is not None:

        story.append(
            Paragraph(
                "4. Trending Topics / Narratives",
                section_style
            )
        )

        topic_data = [
            ["Topic / Entity", "Count"]
        ]

        for item in topics.get(
            "topics",
            []
        ):

            topic_data.append([
                str(
                    item.get(
                        "name",
                        ""
                    )
                ),
                str(
                    item.get(
                        "count",
                        0
                    )
                )
            ])

        if len(topic_data) > 1:

            story.append(
                pdf_table(
                    topic_data,
                    [
                        100 * mm,
                        70 * mm
                    ]
                )
            )

    # INFLUENCE
    if influence is not None:

        story.append(
            Paragraph(
                "5. Influence / Network Analysis",
                section_style
            )
        )

        if influence.get("followers"):

            followers = influence["followers"]

            follower_data = [
                ["Follower Metric", "Value"],
                [
                    "Total Followers",
                    f"{followers.get('total', 0):,.2f}"
                ],
                [
                    "Average Followers",
                    f"{followers.get('average', 0):,.2f}"
                ]
            ]

            story.append(
                pdf_table(
                    follower_data,
                    [
                        100 * mm,
                        70 * mm
                    ]
                )
            )

            story.append(
                Spacer(
                    1,
                    7
                )
            )

        edges = influence.get(
            "edges",
            []
        )

        if edges:

            edge_data = [
                [
                    "Source",
                    "Target",
                    "Mentions"
                ]
            ]

            for edge in edges:

                edge_data.append([
                    str(
                        edge.get(
                            "source",
                            ""
                        )
                    ),
                    str(
                        edge.get(
                            "target",
                            ""
                        )
                    ),
                    str(
                        edge.get(
                            "mentions",
                            0
                        )
                    )
                ])

            story.append(
                Paragraph(
                    "Top Influence Connections",
                    body_style
                )
            )

            story.append(
                pdf_table(
                    edge_data,
                    [
                        62 * mm,
                        62 * mm,
                        46 * mm
                    ]
                )
            )

    story.append(
        Spacer(
            1,
            16
        )
    )

    story.append(
        Paragraph(
            "Generated by SocialPulse AI",
            small_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer


@app.get("/api/reports/{report_id}/pdf")
async def download_report_pdf(
    report_id: str,
    session_id: str = Header(None) # ADDED SESSION ID
):

    if not session_id or session_id not in user_reports:
        raise HTTPException(
            status_code=404,
            detail="Report not found."
        )

    # SEARCH ONLY IN THE USER'S OWN REPORTS
    for report in user_reports[session_id]:

        if report["id"] == report_id:

            pdf_buffer = build_pdf_report(
                report
            )

            filename = (
                f"SocialPulse_Report_"
                f"{report_id[:8]}.pdf"
            )

            return StreamingResponse(
                pdf_buffer,
                media_type="application/pdf",
                headers={
                    "Content-Disposition":
                        f'attachment; filename="{filename}"'
                }
            )

    raise HTTPException(
        status_code=404,
        detail="Report not found."
    )


@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED
    }
