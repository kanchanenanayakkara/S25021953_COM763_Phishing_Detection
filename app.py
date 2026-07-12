import ipaddress
import pickle
import re
from urllib.parse import urlparse

import streamlit as st
import tldextract

from url_feature_extractor import extract_url_features


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Phishing URL Risk Assessment",
    page_icon="🔐",
    layout="centered"
)


# Use the Public Suffix List included with tldextract.
# This prevents the app from downloading the list at runtime.
domain_extractor = tldextract.TLDExtract(
    suffix_list_urls=()
)


# ---------------------------------------------------------
# Load deployment files
# ---------------------------------------------------------

@st.cache_resource
def load_deployment_files():
    """
    Load the trained model, feature order, classification
    threshold and popular-domain reference list.
    """

    with open("url_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("url_features.pkl", "rb") as feature_file:
        feature_order = pickle.load(feature_file)

    with open("url_threshold.pkl", "rb") as threshold_file:
        threshold = pickle.load(threshold_file)

    with open("top_domains.pkl", "rb") as domains_file:
        popular_domains = set(pickle.load(domains_file))

    # Normalise all stored domains
    popular_domains = {
        str(domain).lower().strip().strip(".")
        for domain in popular_domains
    }

    return model, feature_order, threshold, popular_domains


# ---------------------------------------------------------
# URL and domain helper functions
# ---------------------------------------------------------

def normalise_for_parsing(url):
    """
    Add HTTPS when a scheme is missing so urllib can correctly
    identify the hostname.
    """

    if not isinstance(url, str):
        raise TypeError("The URL must be entered as text.")

    cleaned_url = url.strip()

    if not cleaned_url:
        raise ValueError("Please enter a website URL.")

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        cleaned_url
    ):
        cleaned_url = "https://" + cleaned_url

    return cleaned_url


def is_ip_address(hostname):
    """
    Return True when the hostname is an IPv4 or IPv6 address.
    """

    try:
        ipaddress.ip_address(
            hostname.strip("[]")
        )
        return True

    except ValueError:
        return False


def extract_domain_information(url):
    """
    Extract the hostname, registered domain and subdomain.

    Example:
        google.com.evil-site.com

        Hostname:
            google.com.evil-site.com

        Registered domain:
            evil-site.com

        Subdomain:
            google.com
    """

    normalised_url = normalise_for_parsing(url)
    parsed_url = urlparse(normalised_url)

    if parsed_url.scheme.lower() not in {"http", "https"}:
        raise ValueError(
            "Only HTTP and HTTPS website URLs are supported."
        )

    hostname = (
        parsed_url.hostname or ""
    ).lower().strip(".")

    if not hostname:
        raise ValueError(
            "The submitted URL does not contain a valid hostname."
        )

    if any(character.isspace() for character in hostname):
        raise ValueError(
            "The hostname contains invalid spaces."
        )

    # IP addresses do not have a normal public suffix.
    if is_ip_address(hostname):
        registered_domain = hostname
        subdomain = ""

    else:
        extracted = domain_extractor(hostname)

        # A valid public domain should contain both a domain
        # component and a recognised public suffix.
        if not extracted.domain or not extracted.suffix:
            raise ValueError(
                "Please enter a valid public website URL, "
                "for example: https://www.example.com"
            )

        registered_domain = ".".join(
            part
            for part in [
                extracted.domain,
                extracted.suffix
            ]
            if part
        )

        subdomain = extracted.subdomain or ""

    return {
        "hostname": hostname,
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "normalised_url": normalised_url,
        "netloc": parsed_url.netloc
    }


def find_popular_domain_in_subdomain(
    subdomain,
    popular_domains
):
    """
    Detect a recognised popular domain placed deceptively
    inside the subdomain.

    Example:
        google.com.evil-site.com

    The real registered domain is evil-site.com, while
    google.com appears only inside the subdomain.
    """

    if not subdomain:
        return None

    labels = subdomain.lower().split(".")

    for index in range(len(labels)):
        candidate = ".".join(labels[index:])

        if candidate in popular_domains:
            return candidate

    return None


# ---------------------------------------------------------
# Load the files
# ---------------------------------------------------------

try:
    (
        model,
        feature_order,
        phishing_threshold,
        popular_domains
    ) = load_deployment_files()

except FileNotFoundError:
    st.error(
        "A required deployment file is missing. Ensure that "
        "url_model.pkl, url_features.pkl, url_threshold.pkl "
        "and top_domains.pkl are available in the repository."
    )
    st.stop()

except Exception as error:
    st.error(
        f"The deployment files could not be loaded: {error}"
    )
    st.stop()


# ---------------------------------------------------------
# Application interface
# ---------------------------------------------------------

st.title("🔐 Phishing URL Risk Assessment System")

st.write(
    """
    Enter a complete website URL below. The system combines a
    Logistic Regression model with domain-structure and
    popular-domain safety checks.
    """
)

st.info(
    "The application analyses URL text only. It does not open, "
    "download or inspect the submitted website."
)

url_input = st.text_input(
    "Website URL",
    placeholder="https://www.example.com"
)

analyse_button = st.button(
    "Analyse URL",
    type="primary",
    width="stretch"
)


# ---------------------------------------------------------
# URL analysis
# ---------------------------------------------------------

if analyse_button:

    if not url_input.strip():
        st.warning("Please enter a website URL.")

    else:
        try:
            # Generate the same 14 features used for training
            input_features = extract_url_features(
                url_input,
                feature_order
            )

            # Parse the hostname and registered domain
            domain_info = extract_domain_information(
                url_input
            )

            hostname = domain_info["hostname"]

            registered_domain = domain_info[
                "registered_domain"
            ]

            subdomain = domain_info["subdomain"]

            normalised_url = domain_info[
                "normalised_url"
            ]

            netloc = domain_info["netloc"]

            # Generate model probabilities
            probabilities = model.predict_proba(
                input_features
            )[0]

            probability_by_class = dict(
                zip(
                    model.classes_,
                    probabilities
                )
            )

            phishing_score = float(
                probability_by_class.get(0, 0)
            )

            legitimate_score = float(
                probability_by_class.get(1, 0)
            )

            # Apply the saved phishing threshold
            model_flags_phishing = (
                phishing_score >= phishing_threshold
            )

            # Check whether the actual registered domain appears
            # in the popular-domain reference list
            registered_domain_is_popular = (
                registered_domain in popular_domains
            )

            # Detect a recognised brand/domain deceptively placed
            # inside the subdomain
            imitated_popular_domain = (
                find_popular_domain_in_subdomain(
                    subdomain,
                    popular_domains
                )
            )

            # Additional structural safety signals
            ip_address_signal = is_ip_address(
                hostname
            )

            # Detect @ only in the URL authority section.
            # This avoids flagging an email address in a query.
            at_symbol_signal = (
                "@" in netloc
            )

            punycode_signal = (
                "xn--" in hostname
            )

            strong_deception_signal = any([
                ip_address_signal,
                at_symbol_signal,
                imitated_popular_domain is not None
            ])

            st.subheader("Analysis Result")

            # -------------------------------------------------
            # Result 1: Strong deceptive structure
            # -------------------------------------------------

            if strong_deception_signal:
                st.error(
                    "🚨 Result: Likely phishing"
                )

                reasons = []

                if ip_address_signal:
                    reasons.append(
                        "The hostname is an IP address instead "
                        "of a standard registered domain."
                    )

                if at_symbol_signal:
                    reasons.append(
                        "The URL contains an @ symbol in its "
                        "authority section, which can hide the "
                        "real destination."
                    )

                if imitated_popular_domain:
                    reasons.append(
                        f"The subdomain imitates the recognised "
                        f"domain `{imitated_popular_domain}`, "
                        f"while the actual registered domain is "
                        f"`{registered_domain}`."
                    )

                for reason in reasons:
                    st.write(f"- {reason}")

            # -------------------------------------------------
            # Result 2: Model and popular-domain conflict
            # -------------------------------------------------

            elif (
                model_flags_phishing
                and registered_domain_is_popular
            ):
                st.warning(
                    "⚠️ Conflicting signals — manual review required"
                )

                st.write(
                    "The registered domain appears in the "
                    "popular-domain reference list, but the URL "
                    "structure received a high phishing score. "
                    "This can occur with long search, tracking or "
                    "authentication URLs."
                )

            # -------------------------------------------------
            # Result 3: Model indicates phishing
            # -------------------------------------------------

            elif model_flags_phishing:
                st.error(
                    "🚨 Result: Likely phishing"
                )

            # -------------------------------------------------
            # Result 4: Low model risk and recognised domain
            # -------------------------------------------------

            elif registered_domain_is_popular:
                st.success(
                    "✅ Result: Likely legitimate"
                )

            # -------------------------------------------------
            # Result 5: Low model risk but unknown domain
            # -------------------------------------------------

            else:
                st.warning(
                    "⚠️ Unverified domain — manual review recommended"
                )

                st.write(
                    "The model produced a low phishing score, but "
                    "the registered domain was not found in the "
                    "popular-domain reference list."
                )

            # -------------------------------------------------
            # Display model scores
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Model phishing score",
                    f"{phishing_score:.2%}"
                )

            with col2:
                st.metric(
                    "Model legitimate score",
                    f"{legitimate_score:.2%}"
                )

            # -------------------------------------------------
            # Display domain information
            # -------------------------------------------------

            st.write(
                f"**Hostname:** `{hostname}`"
            )

            st.write(
                f"**Registered domain:** "
                f"`{registered_domain}`"
            )

            if subdomain:
                st.write(
                    f"**Subdomain:** `{subdomain}`"
                )

            st.write(
                "**Popular registered-domain signal:** "
                + (
                    "Recognised"
                    if registered_domain_is_popular
                    else "Not recognised"
                )
            )

            # Punycode warning
            if punycode_signal:
                st.warning(
                    "The hostname contains internationalised-domain "
                    "encoding (`xn--`). Review it carefully because "
                    "visually deceptive domain names may use this "
                    "format."
                )

            # Display extracted features
           with st.expander(
    "View extracted URL features"
):
    feature_lines = []

    for feature in feature_order:
        value = input_features.iloc[0][feature]
        feature_lines.append(
            f"{feature}: {value}"
        )

    st.code(
        "\n".join(feature_lines),
        language=None
    )

            st.caption(
                "Scores are machine-learning outputs, not proof "
                "that a website is safe or malicious. Popular-domain "
                "membership and structural checks are supporting "
                "signals only."
            )

        except (ValueError, TypeError) as error:
            st.warning(str(error))

        except Exception as error:
            st.error(
                "An unexpected analysis error occurred: "
                f"{error}"
            )


# ---------------------------------------------------------
# Application information
# ---------------------------------------------------------

st.divider()

st.markdown(
    f"""
    **Model:** Logistic Regression  
    **Input:** 14 lexical and structural URL features  
    **Phishing threshold:** {phishing_threshold:.2f}  
    **Safety checks:** Registered-domain parsing, popular-domain
    conflict detection, deceptive-subdomain detection, IP-address
    detection and authority-section checking  
    **Possible outputs:** Likely legitimate, likely phishing,
    unverified domain or manual review
    """
)
