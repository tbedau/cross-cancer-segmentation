import json
from enum import StrEnum

from sqlmodel import Field, Relationship, SQLModel


class TCGAProject(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str  # Represents the TCGA project identifier, e.g., "SKCM"
    samples: list["Sample"] = Relationship(back_populates="tcga_project")


class Sample(SQLModel, table=True):
    id: int = Field(primary_key=True)
    tcga_project_id: int = Field(foreign_key="tcgaproject.id")
    tcga_project: TCGAProject = Relationship(back_populates="samples")
    tcga_case_id: str
    tcga_sample_id: str | None = Field(default=None)
    tcga_uuid: str | None = Field(default=None)
    slide_mpp: float = Field(description="Microns per pixel measurement")
    tumor_image: str | None = Field(default=None)
    tumor_image_thumbnail: str | None = Field(default=None)
    normal_image: str | None = Field(default=None)
    normal_image_thumbnail: str | None = Field(default=None)
    scoring_tumor: str | None = Field(default=None)
    scoring_normal: str | None = Field(default=None)
    comments_tumor: str | None = Field(default=None)
    comments_normal: str | None = Field(default=None)

    def set_scoring_tumor(self, scoring_data: dict) -> None:
        self.scoring_tumor = json.dumps(scoring_data)

    def get_scoring_tumor(self) -> dict:
        return json.loads(self.scoring_tumor) if self.scoring_tumor else {}

    def set_scoring_normal(self, scoring_data: dict) -> None:
        self.scoring_normal = json.dumps(scoring_data)

    def get_scoring_normal(self) -> dict:
        return json.loads(self.scoring_normal) if self.scoring_normal else {}

    def set_comments_tumor(self, comments_data: dict) -> None:
        self.comments_tumor = json.dumps(comments_data)

    def get_comments_tumor(self) -> dict:
        return json.loads(self.comments_tumor) if self.comments_tumor else {}

    def set_comments_normal(self, comments_data: dict) -> None:
        self.comments_normal = json.dumps(comments_data)

    def get_comments_normal(self) -> dict:
        return json.loads(self.comments_normal) if self.comments_normal else {}


class TissueType(StrEnum):
    tumor = "tumor"
    normal = "normal"
