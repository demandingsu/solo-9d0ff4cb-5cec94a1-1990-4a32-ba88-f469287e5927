"""FastAPI 入口：两区坡地苗木两阶段抽样。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .sampling import (
    District,
    SamplingError,
    SamplingResult,
    SampleEntry,
    Seedling,
    build_sampling_plan,
)
from .schemas import SampleRequest, SampleResponse, SampleEntryModel

app = FastAPI(
    title="两区坡地苗木抽样 API",
    description=(
        "两区首轮独立等概率无放回抽样，未入样坡地苗木合并补抽；"
        "返回入样编号、轮次与整套方案下的入样概率及权重。"
    ),
    version="1.0.0",
)


@app.exception_handler(SamplingError)
async def sampling_error_handler(_: Request, exc: SamplingError) -> JSONResponse:
    payload: dict[str, object] = {"detail": exc.errors}
    if exc.impossible_ids:
        payload["impossible_ids"] = exc.impossible_ids
    return JSONResponse(status_code=422, content=payload)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sample", response_model=SampleResponse)
def create_sample(req: SampleRequest) -> SampleResponse:
    districts = [
        District(
            name=d.name,
            roster=[
                Seedling(id=s.id, is_slope=s.is_slope) for s in d.roster
            ],
            first_round_quota=d.first_round_quota,
            actual_drawn=list(d.actual_drawn),
        )
        for d in req.districts
    ]
    result: SamplingResult = build_sampling_plan(
        districts, req.supplement_quota, req.seed
    )
    return SampleResponse(
        entries=[_entry_model(e) for e in result.entries],
        supplement_pool_size=result.supplement_pool_size,
        supplement_drawn=result.supplement_drawn,
        supplement_quota=result.supplement_quota,
        supplement_all_taken=result.supplement_all_taken,
        inclusion_probabilities={
            name: {"non_slope": p_non, "slope": p_slope}
            for name, (p_non, p_slope) in result.inclusion_probabilities.items()
        },
    )


def _entry_model(entry: SampleEntry) -> SampleEntryModel:
    return SampleEntryModel(
        id=entry.id,
        district=entry.district,
        round_name=entry.round_name,
        inclusion_probability=entry.inclusion_probability,
        weight=entry.weight,
    )
