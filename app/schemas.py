"""请求/响应数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SeedlingModel(BaseModel):
    id: str = Field(..., min_length=1, description="苗木唯一编号")
    is_slope: bool = Field(..., description="是否为坡地苗木")


class DistrictModel(BaseModel):
    name: str = Field(..., min_length=1, description="区名")
    roster: list[SeedlingModel] = Field(..., min_length=1, description="本区完整名册")
    first_round_quota: int = Field(..., ge=0, description="首轮名额（可为 0）")
    actual_drawn: list[str] = Field(
        ..., description="首轮实抽编号（须与名额一致、无重复且属于本区）"
    )


class SampleRequest(BaseModel):
    districts: list[DistrictModel] = Field(
        ..., min_length=2, max_length=2, description="恰好两个区"
    )
    supplement_quota: int = Field(..., ge=0, description="事先定下的补抽名额")
    seed: int = Field(..., description="随机种子（整数，相同输入重跑结果一致）")


class SampleEntryModel(BaseModel):
    id: str
    district: str
    round_name: str = Field(..., description="入样轮次：首轮 / 补抽")
    inclusion_probability: float = Field(
        ..., description="整套方案下该株至少一次入样概率"
    )
    weight: float = Field(..., description="入样概率的倒数")


class SampleResponse(BaseModel):
    entries: list[SampleEntryModel]
    supplement_pool_size: int
    supplement_drawn: int
    supplement_quota: int
    supplement_all_taken: bool
    inclusion_probabilities: dict[str, dict[str, float]]
