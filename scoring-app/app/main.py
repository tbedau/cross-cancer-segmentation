import csv
import io
import json
from collections.abc import Generator
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, SQLModel, col, create_engine, func, select

from app.models import Sample, TCGAProject, TissueType

# Database URL
DATABASE_URL = "sqlite:///./data/samples.db"
engine = create_engine(DATABASE_URL)

# Create tables if they don't exist
SQLModel.metadata.create_all(engine)

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Dependency to get the database session
def get_session() -> Generator[Session]:
    with Session(engine) as session:
        yield session


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/projects")


@app.get("/projects", response_class=HTMLResponse)
def projects_view(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    projects = session.exec(select(TCGAProject)).all()
    project_data = []

    for project in projects:
        total_samples = len(project.samples)
        completely_scored_samples = 0

        for sample in project.samples:
            complete_tumor = not sample.tumor_image

            if sample.tumor_image and sample.scoring_tumor:
                scoring_tumor = json.loads(sample.scoring_tumor)
                complete_tumor = all(score is not None for score in scoring_tumor.values())

            if complete_tumor:
                completely_scored_samples += 1

        progress = (completely_scored_samples / total_samples * 100) if total_samples > 0 else 0

        project_data.append(
            {
                "id": project.id,
                "name": project.name,
                "progress": round(progress, 2),
                "scored_samples": completely_scored_samples,
                "total_samples": total_samples,
            }
        )

    return templates.TemplateResponse(
        "pages/projects.html.jinja",
        {"request": request, "projects": project_data, "title": "Projects"},
    )


@app.get("/project/{project_id}/samples", response_class=HTMLResponse)
def project_samples_view(
    request: Request,
    project_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, ge=1),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    project = session.get(TCGAProject, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    samples_query = (
        select(Sample)
        .where(Sample.tcga_project_id == project_id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    samples = session.exec(samples_query).all()

    total_samples_count_query = select(func.count()).select_from(
        select(Sample).where(Sample.tcga_project_id == project_id).subquery()
    )
    total_samples_count = session.exec(total_samples_count_query).one()

    has_next_page = (page * page_size) < total_samples_count

    return templates.TemplateResponse(
        "pages/project.html.jinja",
        {
            "request": request,
            "project": project,
            "samples": samples,
            "has_next_page": has_next_page,
            "next_page": page + 1,
            "title": f"{project.name} Project Overview",
        },
    )


@app.get("/project/{project_id}/rows", response_class=HTMLResponse)
async def samples_view(
    request: Request,
    project_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, ge=1),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    base_query = select(Sample).where(Sample.tcga_project_id == project_id)

    samples_query = base_query.offset((page - 1) * page_size).limit(page_size)
    samples = session.exec(samples_query).all()

    total_samples = session.exec(select(func.count()).select_from(base_query.subquery())).one()
    has_next_page = (page * page_size) < total_samples

    return templates.TemplateResponse(
        "fragments/sample_rows.html.jinja",
        {
            "request": request,
            "project_id": project_id,
            "samples": samples,
            "total_samples": total_samples,
            "has_next_page": has_next_page,
            "next_page": page + 1,
        },
    )


@app.get("/sample/{sample_id}/{tissue_type}", response_class=HTMLResponse)
def sample_view(
    request: Request,
    sample_id: int,
    tissue_type: TissueType,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    sample = session.get(Sample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    scoring_field = "scoring_tumor" if tissue_type == TissueType.tumor else "scoring_normal"
    scoring = json.loads(getattr(sample, scoring_field)) if getattr(sample, scoring_field) else {}

    comments_field = "comments_tumor" if tissue_type == TissueType.tumor else "comments_normal"
    comments = (
        json.loads(getattr(sample, comments_field)) if getattr(sample, comments_field) else {}
    )

    image_file = sample.tumor_image if tissue_type == TissueType.tumor else sample.normal_image

    samples_in_project = session.exec(
        select(Sample)
        .where(Sample.tcga_project_id == sample.tcga_project_id)
        .order_by(col(Sample.id))
    ).all()
    sample_ids = [s.id for s in samples_in_project]
    current_index = sample_ids.index(sample_id)

    next_sample_id = sample_ids[current_index + 1] if current_index + 1 < len(sample_ids) else None
    prev_sample_id = sample_ids[current_index - 1] if current_index > 0 else None

    return templates.TemplateResponse(
        "pages/sample.html.jinja",
        {
            "request": request,
            "sample": sample,
            "image_file": image_file,
            "scoring": scoring,
            "comments": comments,
            "next_sample_id": next_sample_id,
            "prev_sample_id": prev_sample_id,
            "tissue_type": tissue_type.value,  # Pass the value of the enum to the template
            "title": "Sample View",
        },
    )


@app.post("/sample/{sample_id}/{tissue_type}/score")
async def update_score(
    sample_id: int,
    tissue_type: TissueType,
    modelId: int | None = Form(None),
    rating: str = Form(...),
    session: Session = Depends(get_session),
) -> Response:
    sample = session.get(Sample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    scoring_attr = "scoring_tumor" if tissue_type == TissueType.tumor else "scoring_normal"

    current_scoring = json.loads(getattr(sample, scoring_attr) or "{}")

    if modelId is not None:
        algorithm_key = f"model{modelId}"
        if str(current_scoring.get(algorithm_key)) == rating:
            current_scoring[algorithm_key] = None
        else:
            current_scoring[algorithm_key] = None if rating == "clear" else int(rating)
    else:
        update_score = None if rating == "clear" else int(rating)
        for i in range(1, 6):
            current_scoring[f"model{i}"] = update_score

    setattr(sample, scoring_attr, json.dumps(current_scoring))
    session.add(sample)
    session.commit()

    return Response(status_code=204, headers={"HX-Trigger": "scoresUpdated"})


@app.get("/sample/{sample_id}/{tissue_type}/model/{model_id}/scoring-bar")
async def get_scoring_bar(
    request: Request,
    sample_id: int,
    tissue_type: TissueType,
    model_id: int,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    sample = session.get(Sample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    if tissue_type == TissueType.tumor:
        scoring = sample.get_scoring_tumor() if sample.scoring_tumor else {}
    else:
        scoring = sample.get_scoring_normal() if sample.scoring_normal else {}

    context = {
        "request": request,
        "sample": sample,
        "i": model_id,
        "scoring": scoring,
        "tissue_type": tissue_type.value,
    }

    return templates.TemplateResponse("fragments/scoring_bar.html.jinja", context)


@app.put("/sample/{sample_id}/{tissue_type}/model/{model_id}/comment/edit")
async def update_comment(
    request: Request,
    sample_id: int,
    tissue_type: TissueType,
    model_id: int,
    comment: str | None = Form(None),
    session: Session = Depends(get_session),
) -> dict[str, str]:
    sample = session.get(Sample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    if tissue_type == TissueType.tumor:
        comments = sample.get_comments_tumor() if sample.comments_tumor else {}
        comments[f"model{model_id}"] = comment
        sample.set_comments_tumor(comments)
    else:
        comments = sample.get_comments_normal() if sample.comments_normal else {}
        comments[f"model{model_id}"] = comment
        sample.set_comments_normal(comments)

    session.add(sample)
    session.commit()

    return {"message": "Comment updated successfully"}


CSV_HEADER = [
    "file_name",
    "tcga_project",
    "tcga_case_id",
    "tcga_sample_id",
    "tcga_uuid",
    "slide_mpp",
    "tissue_type",
    "model",
    "score",
    "comment",
]


def _write_sample_rows(writer: Any, sample: Sample, project_name: str) -> None:
    for tissue_type, image_attr, scoring_getter, comments_attr in [
        ("tumor", "tumor_image", sample.get_scoring_tumor, "comments_tumor"),
        ("normal", "normal_image", sample.get_scoring_normal, "comments_normal"),
    ]:
        image = getattr(sample, image_attr)
        if not image:
            continue
        scoring = scoring_getter()
        comments = json.loads(getattr(sample, comments_attr) or "{}")
        for model in range(1, 6):
            model_key = f"model{model}"
            writer.writerow(
                [
                    image.replace("modelx", model_key),
                    project_name,
                    sample.tcga_case_id,
                    sample.tcga_sample_id,
                    sample.tcga_uuid,
                    sample.slide_mpp,
                    tissue_type,
                    model_key,
                    scoring.get(model_key, "NA"),
                    comments.get(model_key, ""),
                ]
            )


def _scoring_csv_response(output: io.StringIO, file_name: str) -> StreamingResponse:
    output.seek(0)
    return StreamingResponse(
        iter([output.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={file_name}"},
    )


@app.get("/download/project/{project_id}")
def download_project_csv(
    project_id: int, session: Session = Depends(get_session)
) -> StreamingResponse:
    project = session.get(TCGAProject, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    samples = session.exec(select(Sample).where(Sample.tcga_project_id == project_id)).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(CSV_HEADER)

    for sample in samples:
        _write_sample_rows(writer, sample, project.name)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return _scoring_csv_response(
        output, f"{timestamp}_{project.name.replace(' ', '_')}_scoring.csv"
    )


@app.get("/download/all")
def download_all_scoring_data(
    session: Session = Depends(get_session),
) -> StreamingResponse:
    projects = session.exec(select(TCGAProject)).all()

    if not projects:
        raise HTTPException(status_code=404, detail="No projects found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(CSV_HEADER)

    for project in projects:
        for sample in project.samples:
            _write_sample_rows(writer, sample, project.name)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return _scoring_csv_response(output, f"{timestamp}_all_projects_scoring.csv")
