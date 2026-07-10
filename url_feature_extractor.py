import ipaddress
import re
from urllib.parse import urlparse

import pandas as pd


def normalise_url(url: str) -> str:
    """
    Validate and normalise a submitted URL.

    A temporary HTTPS scheme is added when the user does not
    provide a scheme, allowing urllib to parse the domain.
    """
    if not isinstance(url, str):
        raise TypeError("The URL must be provided as text.")

    url = url.strip()

    if not url:
        raise ValueError("Please enter a URL.")

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        url
    ):
        url = "https://" + url

    return url


def check_domain_ip(domain: str) -> int:
    """
    Return 1 if the domain is an IP address; otherwise return 0.
    """
    clean_domain = domain.split(":")[0].strip("[]")

    try:
        ipaddress.ip_address(clean_domain)
        return 1
    except ValueError:
        return 0


def extract_url_features(
    url: str,
    feature_order: list[str]
) -> pd.DataFrame:
    """
    Extract the 14 lexical and structural URL features used by
    the trained phishing-detection model.
    """
    normalised_url = normalise_url(url)
    parsed_url = urlparse(normalised_url)

    domain = parsed_url.hostname or ""

    if not domain:
        raise ValueError(
            "The submitted text does not contain a valid domain."
        )

    domain_parts = [
        part for part in domain.split(".")
        if part
    ]

    tld = (
        domain_parts[-1]
        if len(domain_parts) >= 2
        else ""
    )

    number_of_subdomains = max(
        len(domain_parts) - 2,
        0
    )

    # This follows the calculation validated against the dataset.
    url_length = max(
        len(normalised_url) - 1,
        0
    )

    domain_length = len(domain)

    number_of_digits = sum(
        character.isdigit()
        for character in normalised_url
    )

    number_of_equals = normalised_url.count("=")
    number_of_question_marks = normalised_url.count("?")
    number_of_ampersands = normalised_url.count("&")

    obfuscated_sequences = re.findall(
        r"%[0-9A-Fa-f]{2}",
        normalised_url
    )

    number_of_obfuscated_characters = len(
        obfuscated_sequences
    )

    has_obfuscation = int(
        number_of_obfuscated_characters > 0
    )

    features = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": check_domain_ip(domain),
        "TLDLength": len(tld),
        "NoOfSubDomain": number_of_subdomains,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar":
            number_of_obfuscated_characters,
        "ObfuscationRatio": (
            number_of_obfuscated_characters / url_length
            if url_length else 0
        ),
        "NoOfDegitsInURL": number_of_digits,
        "DegitRatioInURL": (
            number_of_digits / url_length
            if url_length else 0
        ),
        "NoOfEqualsInURL": number_of_equals,
        "NoOfQMarkInURL":
            number_of_question_marks,
        "NoOfAmpersandInURL":
            number_of_ampersands,
        "IsHTTPS": int(
            parsed_url.scheme.lower() == "https"
        )
    }

    missing_features = [
        feature
        for feature in feature_order
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            "The feature extractor is missing: "
            + ", ".join(missing_features)
        )

    feature_values = [
        features[feature]
        for feature in feature_order
    ]

    return pd.DataFrame(
        [feature_values],
        columns=feature_order
    )
