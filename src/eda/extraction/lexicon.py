"""Curated, extensible taxonomy for the maintenance free-text fields.

Each :class:`Term` maps a set of regex ``patterns`` (matched against the
accent-folded, lower-cased text — see :mod:`normalize`) to a canonical bilingual
label ``"<German> (<English>)"``.

To extend coverage, **add a Term** to the relevant category list (or add a new
category to :data:`CATEGORIES`). Patterns are plain-ASCII regex fragments; order
matters only for the order matched labels are listed in a cell. Patterns are tried
independently, so one text can yield several labels per category (joined with "; ").

Categories
----------
* ``component``  — the affected machine module / part
* ``phenomenon`` — the defect / symptom (what went wrong)
* ``action``     — the corrective action / reaction the technician took
* ``reason``     — the trigger / root cause, when stated
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    de: str                      # canonical German label
    en: str                      # English gloss (shown in brackets)
    patterns: tuple[str, ...]    # ASCII regex fragments matched on normalized text

    @property
    def label(self) -> str:
        return f"{self.de} ({self.en})"


CATEGORIES = ("component", "phenomenon", "action", "reason")


# --------------------------------------------------------------------------- #
# component — which module / part is involved
# --------------------------------------------------------------------------- #
_COMPONENT = [
    Term("Abstandsmessung", "distance measurement",
         (r"abstand", r"distanzmess", r"keyence", r"check.?the.?checker",
          r"check den checker")),
    Term("Vereinzeler", "singulator",
         (r"vereinzeler", r"hauptvereinzel")),
    Term("Transportband", "conveyor belt",
         (r"transportband", r"foerderband", r"\btransport\b", r"\bauslauf\b",
          r"\beinlauf\b", r"rucklaufband")),
    Term("Transportriemen", "conveyor belt drive",
         (r"transportriemen", r"\briemen\b", r"traverse")),
    Term("Roboter", "robot",
         (r"\brobot", r"\brobi\b")),
    Term("Greifer", "gripper",
         (r"greifer",)),
    Term("Plasmagenerator", "plasma generator",
         (r"plasmagenerato", r"plasma.?generator", r"\bgenerator")),
    Term("Plasmadüse", "plasma nozzle",
         (r"plasmaduse", r"\bduse\b", r"\bdse\b")),
    Term("Plasma", "plasma process",
         (r"\bplasma\b", r"plasmabewegung", r"intermit")),
    Term("Niederhalter", "hold-down",
         (r"niederhalter",)),
    Term("Werkstückträger", "workpiece carrier",
         (r"\bwt\b", r"\bwts\b", r"werkstucktrager", r"\bw t\b",
          r"pruf.?wt", r"\ba wt\b", r"\bb wt\b")),
    Term("Werkstückträgerhalter", "carrier holder",
         (r"werkstucktragerhalter", r"werkstuck trager", r"tragerhalter")),
    Term("Leiterplatte", "PCB",
         (r"\blp\b", r"leiterplatt", r"\bplatten\b", r"leiterplatten")),
    Term("Lötrahmen", "solder frame",
         (r"lotrahmen", r"\blr\b")),
    Term("Laststromkreis", "load circuit",
         (r"laststromkreis",)),
    Term("Sicherheitskreis", "safety circuit",
         (r"sicherheitskreis", r"not.?aus", r"not/aus")),
    Term("Schutztür", "safety door",
         (r"schutztur", r"\btur\b", r"\bturen\b")),
    Term("Sensor", "sensor",
         (r"\bsensor\b",)),
    Term("Scanner", "scanner / reader",
         (r"scanner", r"lesefehler", r"\bewak\b")),
    Term("Dreheinheit", "rotary unit",
         (r"dreheinheit", r"drehposition")),
    Term("Achssystem", "axis system",
         (r"achssystem", r"antriebshardware", r"nc-drv")),
    Term("Steuerung/Task", "control task",
         (r"\btask\b", r"automatik", r"automatikmodus", r"automatikbetrieb")),
]


# --------------------------------------------------------------------------- #
# phenomenon — the defect / symptom
# --------------------------------------------------------------------------- #
_PHENOMENON = [
    Term("niO / fehlerhaft", "not OK / faulty",
         (r"\bnio\b", r"n\.?i\.?o", r"fehlerhaft", r"\bfehler\b", r"defekt")),
    Term("außer Toleranz", "out of tolerance",
         (r"ausserhalb der toleranz", r"ausser.*toleranz",
          r"ausserhalb des gultigen", r"distanz zu hoch")),
    Term("gerissen", "torn / broken",
         (r"gerissen",)),
    Term("verklemmt / blockiert", "jammed / blocked",
         (r"verklemmt", r"verkantet", r"geklemmt", r"klemmen", r"blockiert",
          r"\bstau\b", r"verursacht", r"steckt", r"stecken")),
    Term("Kollision", "collision",
         (r"kollision", r"aufeinander", r"zusammen gefahren", r"ineinander",
          r"aufgefahren")),
    Term("falsche Lage", "incorrect position",
         (r"liegt falsch", r"nicht korrekt", r"nicht richtig", r"\bschief\b",
          r"\bschrag\b", r"versetzt", r"\bquer\b", r"falsch ausgehoben",
          r"aus der spur")),
    Term("ohne Fehlermeldung", "no error message",
         (r"ohne (jegliche )?fehlermeldung", r"keine fehlermeldung",
          r"keine meldung", r"keine st(o|oe)rung")),
    Term("keine Grundstellung", "no home position",
         (r"keine grundstellung", r"grundstellung.*nicht", r"kein grundstellung")),
    Term("nicht stabilisiert", "not stabilized",
         (r"nicht stabilisiert", r"muss stabilisiert", r"stabilisiert werden")),
    Term("Zeitüberschreitung", "timeout",
         (r"zeituberschreitung", r"time.?out", r"timeout",
          r"takt.?zeit passt nicht", r"taktzeit", r"abgelaufen")),
    Term("läuft / fährt nicht", "does not run / move",
         (r"lauft nicht", r"fahrt nicht", r"fahren nicht", r"fuhrt nicht",
          r"lasst sich nicht einschalten", r"lauft.*nicht an",
          r"nicht erreicht", r"nicht hinein", r"nicht rein", r"offnet nicht",
          r"dreht sich nicht", r"verriegelt nicht")),
    Term("Teil gefallen", "part dropped",
         (r"fallen lassen", r"fallt", r"fallen als")),
    Term("nicht erkannt / gelesen", "not detected / read",
         (r"nicht erkannt", r"nicht gelesen", r"nicht best(u|ue)ckt",
          r"erkennen.*nicht", r"\bfehlt\b")),
    Term("Soll-Ist-Abweichung", "setpoint mismatch",
         (r"soll.?=?.?[01]", r"\bsoll\b")),
    Term("Stillstand", "standstill / no message",
         (r"steht ohne", r"steht scheinbar", r"stehen geblieben", r"stillstand",
          r"steht am", r"steht mit", r"leer gefahren")),
    Term("Schleppfehler", "following error",
         (r"schlepp",)),
    Term("Staudruck niO", "back-pressure not OK",
         (r"staudruck", r"bk.?pressure", r"pltbk")),
    Term("Steuerungsfehler", "controller / PLC error",
         (r"errorno", r"interner fehler", r"loc901", r"diagno")),
    Term("Spannungsverlust", "power / voltage loss",
         (r"abgeschaltet", r"deaktiviert", r"spannung", r"versorgungsspannung",
          r"betriebsbereit")),
]


# --------------------------------------------------------------------------- #
# action — corrective action / reaction taken
# --------------------------------------------------------------------------- #
_ACTION = [
    Term("neu eingelegt", "re-inserted",
         (r"neu eingelegt", r"neu eingelegen")),
    Term("Grundstellung gefahren", "homed",
         (r"grundstellung gefahren", r"grundstellungsfahrt", r"\bgs fahrt\b",
          r"grundstellung auto")),
    Term("Ausgangsstellung angefahren", "moved to start position",
         (r"ausgangsstellung",)),
    Term("quittiert", "acknowledged",
         (r"quittiert",)),
    Term("entfernt / entnommen", "removed",
         (r"entfernt", r"entnommen", r"heraus.?gefahren", r"entnahme",
          r"herausgefahren", r"raus")),
    Term("gereinigt", "cleaned",
         (r"gereinigt",)),
    Term("Neustart", "restart",
         (r"neustart", r"neu gestartet", r"neu gestartet", r"\bstart\b.*\bio\b",
          r"wieder aufgebaut")),
    Term("getauscht / umgebaut", "replaced / rebuilt",
         (r"tausch", r"getauscht", r"umbau", r"\bume\b", r"instandsetzung",
          r"instand")),
    Term("Lage korrigiert", "position corrected",
         (r"lage korrigiert", r"gerichtet", r"richtig positioniert",
          r"positioniert", r"korrigiert", r"richtig abgeholt")),
    Term("eingestellt", "adjusted",
         (r"einstellen", r"eingestellt", r"\bsvo\b")),
    Term("an AFA übergeben", "handed to repair (AFA)",
         (r"afa ubergeben", r"\bafa\b", r"rep\.? platz", r"reparatur")),
    Term("überprüft", "inspected",
         (r"uberprufung", r"uberpruft", r"kontrolliert")),
    Term("sonstige Reaktion", "other reaction",
         (r"reaktion: sonstiges", r"\bsonstiges\b")),
]


# --------------------------------------------------------------------------- #
# reason — trigger / root cause (often absent)
# --------------------------------------------------------------------------- #
_REASON = [
    Term("nach Wartung", "after maintenance",
         (r"nach (der )?wartung", r"storung nach wartung")),
    Term("nach Rüsten", "after retooling",
         (r"nach.*rusten", r"hw.?rusten", r"sw.?rusten", r"rusten")),
    Term("wiederkehrende Störung", "recurring fault",
         (r"wiederkehrend",)),
    Term("nach Stillstand", "after standstill",
         (r"langeren stillstand", r"vorschicht", r"langerem stillstand")),
    Term("vorsorglich", "precautionary",
         (r"vorsorglich",)),
]


LEXICON: dict[str, list[Term]] = {
    "component": _COMPONENT,
    "phenomenon": _PHENOMENON,
    "action": _ACTION,
    "reason": _REASON,
}
