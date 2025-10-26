import csv
import io
from importlib.resources import files


def get_ds_phone_dict(dict_name: str, g2p: bool = True) -> dict[str, str]:
    opencpop_dict = {}
    dict_dir = files("ds2aces.ds") / "dicts"
    if (dict_content := (dict_dir / f"{dict_name}.txt").read_text(encoding="utf-8")) is None:
        msg = "Cannot find dict."
        raise FileNotFoundError(msg)
    reader = csv.DictReader(
        io.StringIO(dict_content),
        delimiter="\t",
        fieldnames=["pinyin", "phone"],
    )
    for row in reversed(list(reader)):
        if g2p:
            opencpop_dict[row["pinyin"]] = row["phone"]
        else:
            opencpop_dict[row["phone"]] = row["pinyin"]
    return opencpop_dict


def get_vowels_set(dict_name: str) -> set[str]:
    vowels_set = set()
    dict_dir = files("ds2aces.ds") / "dicts"
    if (dict_content := (dict_dir / f"{dict_name}.txt").read_text(encoding="utf-8")) is None:
        msg = "Cannot find dict."
        raise FileNotFoundError(msg)
    reader = csv.DictReader(
        io.StringIO(dict_content),
        delimiter="\t",
        fieldnames=["pinyin", "phone"],
    )
    for row in reader:
        phones = row["phone"].split()
        if len(phones) > 1:
            vowels_set.add(phones[0])
    return vowels_set
