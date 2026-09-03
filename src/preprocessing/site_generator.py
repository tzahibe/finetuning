from src.datasets.schema import Site
from src.preprocessing.boomi_parser import ParsedBoomiRecord


def generate_site(record: ParsedBoomiRecord) -> Site:
    return Site(
        width_m=record.site_width_m,
        length_m=record.site_length_m,
        area_m2=record.site_area_m2,
    )
