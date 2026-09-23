"""
Mapping from raw source filenames (investing.com export naming) to a
canonical Symbol and best-effort human-readable Company name.

IMPORTANT:
The raw filenames come from investing.com exports. Codes ending in
"c1_NS" are that site's naming for continuous NSE futures-contract
series, not the plain equity ticker. A handful of codes (INGL, MAXE)
are ambiguous and could not be identified with confidence — they are
kept as-is and flagged. The `Symbol` used throughout this pipeline is
always the original file code, so no data is ever mis-attributed even
if a `Company` display name guess is imprecise. Verify uncertain names
before relying on them in later sprints.
"""

CODE_MAP = {
    "ADELc1_NS": ("Adani Enterprises (Futures)", "ADELc1_NS"),
    "APLH": ("Apollo Hospitals", "APLH"),
    "APSEc1_NS": ("Adani Ports & SEZ (Futures)", "APSEc1_NS"),
    "ASPN": ("Asian Paints", "ASPN"),
    "AXBK": ("Axis Bank", "AXBK"),
    "BAJAc1_NS": ("Bajaj Auto (Futures)", "BAJAc1_NS"),
    "BAJE": ("Bajaj Finserv", "BAJE"),
    "BJFN": ("Bajaj Finance", "BJFN"),
    "BJFS": ("Bajaj Finserv (Alt code)", "BJFS"),
    "BRTI": ("Bharti Airtel", "BRTI"),
    "CIPL": ("Cipla", "CIPL"),
    "COAL": ("Coal India", "COAL"),
    "EICH": ("Eicher Motors", "EICH"),
    "ETEA": ("Eternal (formerly Zomato)", "ETEA"),
    "GRAS": ("Grasim Industries", "GRAS"),
    "HALC": ("Hindalco Industries", "HALC"),
    "HCLT": ("HCL Technologies", "HCLT"),
    "HDBKc1_NS": ("HDFC Bank (Futures)", "HDBKc1_NS"),
    "HDFLc1_NS": ("HDFC Life Insurance (Futures)", "HDFLc1_NS"),
    "HLL": ("Hindustan Unilever", "HLL"),
    "ICBK": ("ICICI Bank", "ICBK"),
    "INFY": ("Infosys", "INFY"),
    "INGL": ("UNVERIFIED - Indian Oil Corp (alt code, unconfirmed)", "INGL"),
    "ITC": ("ITC Limited", "ITC"),
    "JIOF": ("Jio Financial Services", "JIOF"),
    "JSTL": ("JSW Steel", "JSTL"),
    "KTKMc1_NS": ("Kotak Mahindra Bank (Futures)", "KTKMc1_NS"),
    "LART": ("Larsen & Toubro", "LART"),
    "MAHM": ("Mahindra & Mahindra", "MAHM"),
    "MAXE": ("UNVERIFIED - unconfirmed underlying", "MAXE"),
    "MRTI": ("Maruti Suzuki", "MRTI"),
    "NEST": ("Nestle India", "NEST"),
    "NTPCc1_NS": ("NTPC (Futures)", "NTPCc1_NS"),
    "ONGC": ("Oil & Natural Gas Corporation", "ONGC"),
    "PGRDc1_NS": ("Power Grid Corporation (Futures)", "PGRDc1_NS"),
    "REDY": ("Dr. Reddy's Laboratories", "REDY"),
    "RELI": ("Reliance Industries", "RELI"),
    "SBIc1_NS": ("State Bank of India (Futures)", "SBIc1_NS"),
    "SHMF": ("Shriram Finance", "SHMF"),
    "SUN": ("Sun Pharmaceutical Industries", "SUN"),
    "TACN": ("Tata Consumer Products", "TACN"),
    "TAMO": ("Tata Motors [renamed Tata Motors Passenger Vehicles Ltd (TMPV), Oct 2025]", "TAMO"),
    "TAMOc1_NS": ("Tata Motors (Futures) [now TMPV]", "TAMOc1_NS"),
    "TCSc1_NS": ("Tata Consultancy Services (Futures)", "TCSc1_NS"),
    "TEML": ("Tech Mahindra", "TEML"),
    "TISC": ("Tata Steel", "TISC"),
    "TITN": ("Titan Company", "TITN"),
    "TREN": ("Trent Limited", "TREN"),
    "ULTC": ("UltraTech Cement", "ULTC"),
    "WIPR": ("Wipro", "WIPR"),
}


def code_from_filename(filename: str) -> str:
    """Extract the raw source code from a 'XXXX Historical Data.csv' filename."""
    return filename.replace(" Historical Data.csv", "").strip()


def company_for_code(code: str):
    """Return (company_name, symbol) for a given raw code, defaulting to the code itself."""
    return CODE_MAP.get(code, (f"UNKNOWN ({code})", code))
