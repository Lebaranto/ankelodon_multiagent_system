from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
import platform, sys

class Limits(BaseModel):
    timeout_seconds: int = Field(12, ge=1, le=120)
    max_stdout_chars: int = Field(10000, ge=256, le=200_000)
    max_stderr_chars: int = Field(10000, ge=256, le=200_000)
    max_plots: int = Field(4, ge=0, le=10)
    max_dataframes: int = Field(3, ge=0, le=10)
    max_df_rows: int = Field(20, ge=1, le=200)
    max_df_cols: int = Field(20, ge=1, le=200)
    plot_dpi: int = Field(120, ge=72, le=300)
    max_pixels: int = Field(25_000_000, ge=1)  # если вдруг юзер генерит большие изображения

class CodeRunRequest(BaseModel):
    language: Literal["python"] = "python"
    code: str
    # Явный allowlist модулей (верхнеуровневые имена)
    allowed_modules: List[str] = Field(
        default_factory=lambda: [
            "math","random","statistics","datetime","re","json","fractions","decimal",
            "numpy","pandas","cmath","matplotlib","matplotlib.pyplot", "seaborn","sklearn","sklearn.datasets","sklearn.model_selection", "sympy"
        ]
    )
    # Флаги, что возвращать
    return_plots: bool = True
    return_dataframes: bool = True
    # Ограничения
    limits: Limits = Field(default_factory=Limits)

class PlotArtifact(BaseModel):
    data_base64: str
    format: Literal["png"] = "png"

class DataFrameArtifact(BaseModel):
    name: str
    head: List[Dict[str, Any]]
    shape: List[int]
    dtypes: Dict[str, str]

class EnvInfo(BaseModel):
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    numpy: Optional[str] = None
    pandas: Optional[str] = None
    platform: str = Field(default_factory=platform.platform)

class CodeRunResult(BaseModel):
    execution_id: str
    status: Literal["success","error","timeout"]
    stdout: str = ""
    stderr: str = ""
    result_repr: Optional[str] = None
    plots: List[PlotArtifact] = Field(default_factory=list)
    dataframes: List[DataFrameArtifact] = Field(default_factory=list)
    env: EnvInfo