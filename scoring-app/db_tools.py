"""CLI helpers for initialising and cleaning the scoring database."""

import os
import re
import sys

from dotenv import load_dotenv
from PIL import Image
from sqlmodel import Session, SQLModel, create_engine, delete, select
from tqdm import tqdm

from app.models import Sample, TCGAProject

load_dotenv()

Image.MAX_IMAGE_PIXELS = 200000000
DATABASE_URL = "sqlite:///./data/samples.db"
engine = create_engine(DATABASE_URL)


def parse_filename(
    filename: str,
) -> tuple[str, str, str | None, str | None, str, str] | None:
    """Parse a TCGA segmentation image filename.

    Expects the pattern
    ``TCGA-<PROJECT>-<CASE>[<SAMPLE>][.<UUID>]_<MPP>_<tissue>_model<N>.jpg``.

    Returns ``(project, case_id, sample_id, uuid, mpp, tissue_type)`` or
    ``None`` if *filename* does not match the expected convention.
    """
    pattern1 = (
        r"^TCGA-([A-Z]+)-([A-Z0-9]{2}-[A-Z0-9]{4})(.*?)"
        r"(?:\.([a-fA-F0-9\-]+))?_(\d+\.\d+)_(tumor|normal)_model[1-5]\.jpg$"
    )
    match = re.match(pattern1, filename)

    if match:
        tcga_project_match = match.group(1)
        tcga_project = "TCGA-" + tcga_project_match
        tcga_case_id_match = match.group(2)
        tcga_case_id = "TCGA-" + tcga_case_id_match
        tcga_sample_id_match = match.group(3) if match.group(3) else None
        tcga_sample_id = tcga_case_id + tcga_sample_id_match if tcga_sample_id_match else None
        tcga_uuid = match.group(4) if match.group(4) else None
        slide_mpp = match.group(5)
        tissue_type = match.group(6)

        return (
            tcga_project,
            tcga_case_id,
            tcga_sample_id,
            tcga_uuid,
            slide_mpp,
            tissue_type,
        )
    else:
        return None


def create_thumbnail(
    image_path: str, thumbnail_path: str, size: tuple[int, int] = (256, 256)
) -> None:
    """Create a thumbnail of the image at *image_path* (default 256 x 256 px)."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size)
            img.save(thumbnail_path)
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")


def initialize_database() -> None:
    """Scan ``IMG_DIRECTORY``, create thumbnails, and populate the database.

    For every segmentation image found, the corresponding
    :class:`~app.models.TCGAProject` and :class:`~app.models.Sample` rows are
    created or updated with initial empty scoring dicts.
    """
    img_directory = os.getenv("IMG_DIRECTORY")
    thumbnails_directory = "app/static/img/thumbnails"

    if not os.path.exists(thumbnails_directory):
        os.makedirs(thumbnails_directory)

    with Session(engine) as session:
        SQLModel.metadata.create_all(engine)  # Ensure tables are created

        projects_dict = {}

        pattern2 = re.compile(r"_model[1-5]\.jpg$")

        for filename in tqdm(os.listdir(img_directory), desc="Processing images"):
            if pattern2.search(filename):
                parsed = parse_filename(filename)
                if not parsed:
                    continue
                (
                    tcga_project,
                    tcga_case_id,
                    tcga_sample_id,
                    tcga_uuid,
                    slide_mpp,
                    tissue_type,
                ) = parsed

                # Ensure the TCGAProject exists
                project = projects_dict.get(tcga_project)
                if not project:
                    project = session.exec(
                        select(TCGAProject).where(TCGAProject.name == tcga_project)
                    ).first()
                    if not project:
                        project = TCGAProject(name=tcga_project)
                        session.add(project)
                        session.commit()
                    projects_dict[tcga_project] = project

                existing_sample = session.exec(
                    select(Sample).where(
                        Sample.tcga_project_id == project.id,
                        Sample.tcga_case_id == tcga_case_id,
                        Sample.tcga_sample_id == tcga_sample_id,
                        Sample.slide_mpp == slide_mpp,
                    )
                ).first()

                if not existing_sample:
                    # Create a new sample
                    existing_sample = Sample(
                        tcga_project_id=project.id,
                        tcga_case_id=tcga_case_id,
                        tcga_sample_id=tcga_sample_id,
                        tcga_uuid=tcga_uuid,
                        slide_mpp=slide_mpp,
                    )
                    session.add(existing_sample)

                thumbnail_filename = f"thumbnail_{filename}"

                if tissue_type == "tumor":
                    updated_filename = pattern2.sub("_modelx.jpg", filename)
                    existing_sample.tumor_image = updated_filename
                    existing_sample.tumor_image_thumbnail = thumbnail_filename

                    if not existing_sample.scoring_tumor:
                        existing_sample.set_scoring_tumor({f"model{i}": None for i in range(1, 6)})
                    if not existing_sample.comments_tumor:
                        existing_sample.set_comments_tumor(
                            {f"model{i}": None for i in range(1, 6)}
                        )
                elif tissue_type == "normal":
                    updated_filename = pattern2.sub("_modelx.jpg", filename)
                    existing_sample.normal_image = updated_filename
                    existing_sample.normal_image_thumbnail = thumbnail_filename

                    if not existing_sample.scoring_normal:
                        existing_sample.set_scoring_normal(
                            {f"model{i}": None for i in range(1, 6)}
                        )
                    if not existing_sample.comments_normal:
                        existing_sample.set_comments_normal(
                            {f"model{i}": None for i in range(1, 6)}
                        )

        session.commit()
        print("Database initialization complete.")


def clean_database() -> None:
    """Delete all ``Sample`` and ``TCGAProject`` rows from the database."""
    with Session(engine) as session:
        session.execute(delete(Sample))  # type: ignore[deprecated]
        session.execute(delete(TCGAProject))  # type: ignore[deprecated]
        session.commit()
    print("Database cleaned. All entries deleted.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify an action: 'init' to initialize or 'clean' to clean the database.")
    elif sys.argv[1] == "init":
        initialize_database()
    elif sys.argv[1] == "clean":
        clean_database()
    else:
        print("Invalid action. Use 'init' to initialize or 'clean' to clean the database.")
