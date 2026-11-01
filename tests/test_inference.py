import pytest
from src.models.inference import clean_hallucinated_tail


def test_clean_hallucinated_tail_removes_foreign_scripts():
    hallucinated = (
        "Xin chào tất cả quý vị và các bạn, cùng đến với chương trình tổ chức 247. "
        "Superprises: Increasing innovation, innova strategic,  challenges,ละคร strategic,ละคร adaptement,ละครรางวัล."
    )
    src_text = "Xin chao tat ca quy vi va cac ban den voi chuong trinh to chuc 247"
    cleaned = clean_hallucinated_tail(hallucinated, input_text=src_text)

    # Không còn chữ tiếng Thái ละคร
    assert "ละคร" not in cleaned
    assert "Xin chào tất cả quý vị và các bạn" in cleaned
    # Đã cắt bỏ đuôi thừa quá dài
    assert len(cleaned.split()) <= len(src_text.split()) * 1.5 + 5


def test_clean_hallucinated_tail_normal_sentence():
    normal = "Hôm nay thời tiết rất đẹp, chúng ta cùng đi dạo phố nhé."
    cleaned = clean_hallucinated_tail(normal, input_text="hom nay thoi tiet rat dep chung ta cung di dao pho nhe")
    assert cleaned == normal
