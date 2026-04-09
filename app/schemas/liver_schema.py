from pydantic import BaseModel, Field

class LiverInput(BaseModel):
    age_of_the_patient: int
    gender_of_the_patient: str
    total_bilirubin: float
    direct_bilirubin: float
    alkphos_alkaline_phosphotase: float
    sgpt_alamine_aminotransferase: float
    sgot_aspartate_aminotransferase: float
    total_protiens: float
    alb_albumin: float

    # 🔥 ACCEPT SLASH INPUT FROM POSTMAN
    a_g_ratio_albumin_and_globulin_ratio: float = Field(
        ..., alias="a/g_ratio_albumin_and_globulin_ratio"
    )

    class Config:
        populate_by_name = True