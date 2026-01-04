// AgroIntelligence Dashboard JavaScript
// Translations for bilingual support
const translations = {
    en: {
        tagline: "Farmer-first intelligence",
        appTitle: "AgroIntelligence Dashboard",
        appSubtitle: "Smart crop & fertilizer recommendations with manual and auto modes, weather, schemes, and chatbot support.",
        languageLabel: "Language",
        manualMode: "📝 Manual Mode",
        autoMode: "🤖 Auto Mode",
        modeHint: "Switch between manual soil inputs and auto-filled district profiles.",
        manualHeading: "Manual soil entry",
        manualDescription: "Provide local soil readings (NPK, pH) to fine-tune predictions.",
        districtLabel: "District",
        mandalLabel: "Mandal",
        seasonLabel: "Season",
        soilTypeLabel: "Soil Type",
        waterLabel: "Water Source",
        soilPhLabel: "Soil pH",
        organicLabel: "Organic Carbon (%)",
        nitrogenLabel: "Soil N (kg/ha)",
        phosphorusLabel: "Soil P (kg/ha)",
        potassiumLabel: "Soil K (kg/ha)",
        manualSubmit: "Get manual recommendations",
        autoHeading: "Auto mode (district intelligence)",
        autoDescription: "Just pick a district; soil and climate defaults populate automatically.",
        autoSubmit: "Get auto recommendations",
        loadingText: "Calculating recommendations...",
        locationHeading: "Location snapshot",
        rainfallLabel: "Rainfall (mm)",
        humidityLabel: "Humidity (%)",
        weatherHeading: "Real-time weather",
        recommendationHeading: "Top crop recommendations",
        guidanceHeading: "Primary crop guidance",
        fertilizerPlan: "Fertilizer plan",
        irrigationPlan: "Irrigation plan",
        marketSignal: "Market signal",
        schemesHeading: "Govt. schemes & subsidies",
        chatbotHeading: "Ask Agro AI",
        chatSend: "Send",
    },
    te: {
        tagline: "రైతుల కోసం రూపొందిన మేధస్సు",
        appTitle: "అగ్రోఇంటెలిజెన్స్ డాష్‌బోర్డ్",
        appSubtitle: "స్మార్ట్ పంట & ఎరువుల సూచనలు, ఆటో మోడ్, వాతావరణం, పథకాలు, చాట్‌బాట్ సహాయం.",
        languageLabel: "భాష",
        manualMode: "📝 మాన్యువల్ మోడ్",
        autoMode: "🤖 ఆటో మోడ్",
        modeHint: "మాన్యువల్ ఇన్‌పుట్స్ లేదా జిల్లా ఆధారిత ఆటో ప్రొఫైల్స్‌ను ఎంచుకోండి.",
        manualHeading: "మాన్యువల్ మట్టిమాపు నమోదు",
        manualDescription: "స్థానిక NPK, pH విలువలు నమోదు చేసి మరింత ఖచ్చితత్వం పొందండి.",
        districtLabel: "జిల్లా",
        mandalLabel: "మండలం",
        seasonLabel: "ఋతువు",
        soilTypeLabel: "మట్టి రకం",
        waterLabel: "నీటి వనరు",
        soilPhLabel: "మట్టి pH",
        organicLabel: "సేంద్రీయ కార్బన్ (%)",
        nitrogenLabel: "నైట్రోజన్ (కిలో/హె)",
        phosphorusLabel: "ఫాస్ఫరస్ (కిలో/హె)",
        potassiumLabel: "పొటాషియం (కిలో/హె)",
        manualSubmit: "మాన్యువల్ సూచనలు పొందండి",
        autoHeading: "ఆటో మోడ్ (జిల్లా డేటా)",
        autoDescription: "జిల్లాను ఎంచుకుంటే అవసరమైన మట్టి, వాతావరణ విలువలు ఆటోమేటిక్‌గా వస్తాయి.",
        autoSubmit: "ఆటో సూచనలు పొందండి",
        loadingText: "పంట సూచనలు సిద్ధమవుతున్నాయి...",
        locationHeading: "ప్రాంత వివరాలు",
        rainfallLabel: "వర్షపాతం (మి.మి)",
        humidityLabel: "ఆర్ద్రత (%)",
        weatherHeading: "తాజా వాతావరణం",
        recommendationHeading: "ఉత్తమ పంట సూచనలు",
        guidanceHeading: "ప్రధాన పంట మార్గదర్శకం",
        fertilizerPlan: "ఎరువు ప్రణాళిక",
        irrigationPlan: "పారుదల ప్రణాళిక",
        marketSignal: "మార్కెట్ సంకేతం",
        schemesHeading: "ప్రభుత్వ పథకాలు",
        chatbotHeading: "అగ్రో AIని అడగండి",
        chatSend: "పంపండి",
    },
};

// Application state
const state = {
    language: "en",
    districts: [],
    mandals: {},
    autoDefaults: null,
};

// DOM elements
const elements = {
    manualForm: document.getElementById("manual-form"),
    autoForm: document.getElementById("auto-form"),
    manualTab: document.getElementById("manual-tab"),
    autoTab: document.getElementById("auto-tab"),
    manualDistrict: document.getElementById("manual-district"),
    manualMandal: document.getElementById("manual-mandal"),
    manualSeason: document.getElementById("manual-season"),
    manualSoilType: document.getElementById("manual-soil-type"),
    manualWater: document.getElementById("manual-water"),
    manualSoilPh: document.getElementById("manual-soil-ph"),
    manualOrganic: document.getElementById("manual-organic"),
    manualN: document.getElementById("manual-n"),
    manualP: document.getElementById("manual-p"),
    manualK: document.getElementById("manual-k"),
    autoDistrict: document.getElementById("auto-district"),
    autoSeason: document.getElementById("auto-season"),
    autoWater: document.getElementById("auto-water"),
    loading: document.getElementById("loading-state"),
    error: document.getElementById("error-state"),
    results: document.getElementById("results"),
    recommendationList: document.getElementById("recommendation-list"),
    location: {
        district: document.getElementById("result-district"),
        mandal: document.getElementById("result-mandal"),
        season: document.getElementById("result-season"),
        soilType: document.getElementById("result-soil-type"),
        rainfall: document.getElementById("result-rainfall"),
        humidity: document.getElementById("result-humidity"),
    },
    weather: document.getElementById("weather-summary"),
    primaryPill: document.getElementById("primary-crop-pill"),
    fertilizer: document.getElementById("fertilizer-plan"),
    irrigation: document.getElementById("irrigation-plan"),
    market: document.getElementById("market-insight"),
    schemeList: document.getElementById("scheme-list"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    chatLog: document.getElementById("chat-log"),
};

// Apply translations to the page
function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((node) => {
        const key = node.getAttribute("data-i18n");
        const translation = translations[state.language][key];
        if (translation) {
            node.textContent = translation;
        }
    });
}

// Set language and update UI
function setLanguage(lang) {
    state.language = lang;
    const enBtn = document.getElementById("lang-en");
    const teBtn = document.getElementById("lang-te");

    if (lang === "en") {
        enBtn.classList.add("bg-green-600", "text-white");
        enBtn.classList.remove("text-green-600");
        teBtn.classList.remove("bg-green-600", "text-white");
        teBtn.classList.add("text-slate-500");
    } else {
        teBtn.classList.add("bg-green-600", "text-white");
        teBtn.classList.remove("text-slate-500");
        enBtn.classList.remove("bg-green-600", "text-white");
        enBtn.classList.add("text-green-600");
    }

    applyTranslations();
}

// Fetch JSON from API
async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || "Request failed");
    }
    return response.json();
}

// Load districts from API
async function loadDistricts() {
    const districts = await fetchJson("/get_district_names");
    state.districts = districts;
    elements.manualDistrict.innerHTML = '<option value="" disabled selected>Choose</option>';
    elements.autoDistrict.innerHTML = '<option value="" disabled selected>Choose</option>';
    districts.forEach((district) => {
        const option = document.createElement("option");
        option.value = district;
        option.textContent = district;
        elements.manualDistrict.appendChild(option.cloneNode(true));
        elements.autoDistrict.appendChild(option);
    });
}

// Handle district change
async function handleDistrictChange(district, mode) {
    if (!district) return;
    try {
        const data = await fetchJson(`/get_district_data/${district}`);
        state.mandals[district] = data.mandals || [];
        if (mode === "manual") {
            populateMandalOptions(district);
            if (data.soil_type) elements.manualSoilType.value = data.soil_type;
            if (data.season) elements.manualSeason.value = data.season;
            elements.manualWater.value = data.water_source || "Tank";
        }
        if (mode === "auto") {
            elements.autoWater.value = data.water_source || "";
        }
    } catch (err) {
        console.error(err);
    }
}

// Populate mandal options
function populateMandalOptions(district) {
    const mandals = state.mandals[district] || [];
    elements.manualMandal.innerHTML = "";
    mandals.forEach((mandal) => {
        const option = document.createElement("option");
        option.value = mandal;
        option.textContent = mandal;
        elements.manualMandal.appendChild(option);
    });
}

// Refresh auto mode defaults
async function refreshAutoDefaults() {
    const district = elements.autoDistrict.value;
    if (!district) return;
    const params = new URLSearchParams({ district });
    if (elements.autoSeason.value) params.append("season", elements.autoSeason.value);

    try {
        const defaults = await fetchJson(`/auto_defaults?${params.toString()}`);
        state.autoDefaults = defaults;
        document.getElementById("auto-soil-ph").textContent = defaults.soil_ph ?? "--";
        document.getElementById("auto-organic").textContent = defaults.organic_carbon ?? "--";
        document.getElementById("auto-n").textContent = defaults.soil_n ?? "--";
        document.getElementById("auto-p").textContent = defaults.soil_p ?? "--";
        document.getElementById("auto-k").textContent = defaults.soil_k ?? "--";
    } catch (error) {
        console.warn(error.message);
    }
}

// Toggle between manual and auto mode
function toggleMode(mode) {
    if (mode === "manual") {
        elements.manualForm.classList.remove("hidden");
        elements.autoForm.classList.add("hidden");
        elements.manualTab.classList.add("tab-active");
        elements.manualTab.classList.remove("tab-idle");
        elements.autoTab.classList.add("tab-idle");
        elements.autoTab.classList.remove("tab-active");
    } else {
        elements.manualForm.classList.add("hidden");
        elements.autoForm.classList.remove("hidden");
        elements.autoTab.classList.add("tab-active");
        elements.autoTab.classList.remove("tab-idle");
        elements.manualTab.classList.add("tab-idle");
        elements.manualTab.classList.remove("tab-active");
    }
}

// Show/hide loading state
function showLoading(show) {
    elements.loading.classList.toggle("hidden", !show);
}

// Show error message
function showError(message) {
    if (!message) {
        elements.error.classList.add("hidden");
        elements.error.textContent = "";
        return;
    }
    elements.error.textContent = message;
    elements.error.classList.remove("hidden");
}

// Render crop recommendations
function renderRecommendations(recommendations) {
    elements.recommendationList.innerHTML = "";
    if (!recommendations.length) {
        elements.recommendationList.innerHTML = "<p class='text-slate-500'>No recommendations available.</p>";
        return;
    }
    recommendations.forEach((rec, index) => {
        const wrapper = document.createElement("div");
        wrapper.className = "flex items-center justify-between bg-slate-50 rounded-2xl p-4";
        const title = document.createElement("div");
        title.innerHTML = `<p class="text-lg font-semibold text-slate-800">${index + 1}. ${rec.crop}</p>`;
        const badge = document.createElement("span");
        badge.className = "text-sm font-semibold text-green-700 bg-green-100 px-3 py-1 rounded-full";
        badge.textContent = `${(rec.score * 100).toFixed(1)}%`;
        wrapper.appendChild(title);
        wrapper.appendChild(badge);
        elements.recommendationList.appendChild(wrapper);
    });
}

// Render guidance information
function renderGuidance(guidance, primaryCrop) {
    elements.primaryPill.textContent = primaryCrop || "--";
    const fertilizerPlan = guidance?.fertilizer_plan;
    const irrigationPlan = guidance?.irrigation_plan;
    const marketIndex = guidance?.market_index;

    elements.fertilizer.innerHTML = fertilizerPlan
        ? `
        <p>N deficit: ${fertilizerPlan.N_deficit_kg_ha ?? "--"} kg/ha</p>
        <p>P deficit: ${fertilizerPlan.P_deficit_kg_ha ?? "--"} kg/ha</p>
        <p>K deficit: ${fertilizerPlan.K_deficit_kg_ha ?? "--"} kg/ha</p>
        <p>Urea: ${fertilizerPlan.urea_kg_ha ?? "--"} kg/ha</p>
        <p>DAP: ${fertilizerPlan.dap_kg_ha ?? "--"} kg/ha</p>
        <p>MOP: ${fertilizerPlan.mop_kg_ha ?? "--"} kg/ha</p>
      `
        : "<p class='text-slate-500'>No fertilizer plan available.</p>";

    elements.irrigation.innerHTML = irrigationPlan
        ? `
        <p>Method: ${irrigationPlan.method ?? "--"}</p>
        <p>Need: ${irrigationPlan.seasonal_need_mm ?? "--"} mm</p>
        <p>Weekly: ${irrigationPlan.mm_per_week ?? "--"} mm</p>
      `
        : "<p class='text-slate-500'>No irrigation plan available.</p>";

    const marketValid = typeof marketIndex === "number" && !Number.isNaN(marketIndex);
    elements.market.textContent = marketValid
        ? `Price index: ${(marketIndex * 100).toFixed(0)}%`
        : "Market signal unavailable.";
}

// Update weather card
function updateWeatherCard(weather) {
    const current = weather && weather.current ? weather.current : weather;
    if (!current) {
        elements.weather.textContent = "Weather feed unavailable.";
        return;
    }
    const timestamp = current.time ? new Date(current.time).toLocaleString() : "";
    elements.weather.textContent = `Temp: ${current.temperature ?? "--"}°C | Wind: ${current.windspeed ?? "--"} km/h${timestamp ? " | Updated: " + timestamp : ""}. `;
    elements.weather.textContent += "Explore the Weather page for hourly trends.";
}

// Render location snapshot
function renderLocation(snapshot) {
    elements.location.district.textContent = snapshot.district || "--";
    elements.location.mandal.textContent = snapshot.mandal || "--";
    elements.location.season.textContent = snapshot.season || "--";
    elements.location.soilType.textContent = snapshot.soil_type || "--";
    elements.location.rainfall.textContent = snapshot.rainfall ?? "--";
    elements.location.humidity.textContent = snapshot.humidity ?? "--";
}

// Submit recommendation request
async function submitRecommendation(mode) {
    showError("");
    showLoading(true);
    elements.results.classList.add("hidden");

    // Scroll to results section smoothly
    setTimeout(() => {
        elements.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 500);

    let payload = { mode };

    try {
        if (mode === "manual") {
            if (!elements.manualDistrict.value) throw new Error("Please select a district.");
            payload = {
                ...payload,
                district: elements.manualDistrict.value,
                mandal: elements.manualMandal.value,
                season: elements.manualSeason.value,
                soil_type: elements.manualSoilType.value,
                water_source: elements.manualWater.value,
                soil_ph: elements.manualSoilPh.value,
                organic_carbon: elements.manualOrganic.value,
                soil_n: elements.manualN.value,
                soil_p: elements.manualP.value,
                soil_k: elements.manualK.value,
            };
        } else {
            if (!elements.autoDistrict.value) throw new Error("Please select a district.");
            payload = {
                ...payload,
                district: elements.autoDistrict.value,
                season: elements.autoSeason.value || null,
                water_source: elements.autoWater.value || null,
            };
        }

        const response = await fetchJson("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        renderLocation(response.location_details || {});
        renderRecommendations(response.recommendations || []);
        renderGuidance(response.guidance, response.recommendations?.[0]?.crop);
        updateWeatherCard(response.weather);
        elements.results.classList.remove("hidden");
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// Load government schemes
async function loadSchemes() {
    try {
        const schemes = await fetchJson("/schemes");
        elements.schemeList.innerHTML = "";
        schemes.forEach((scheme) => {
            const item = document.createElement("li");
            item.innerHTML = `
        <p class="font-semibold text-slate-800">${scheme.title}</p>
        <p class="text-slate-500">${scheme.description}</p>
        <a class="text-green-600 text-sm font-semibold" href="${scheme.link}" target="_blank">More</a>
      `;
            elements.schemeList.appendChild(item);
        });
    } catch (error) {
        elements.schemeList.innerHTML = "<li class='text-slate-500'>Unable to load schemes.</li>";
    }
}

// Append chat message
function appendChatMessage(sender, message) {
    const bubble = document.createElement("div");
    bubble.className = sender === "user" ? "text-right" : "text-left";
    bubble.innerHTML = `
    <span class="inline-block px-3 py-2 rounded-2xl ${sender === "user" ? "bg-green-600 text-white" : "bg-slate-200 text-slate-800"
        }">${message}</span>
  `;
    elements.chatLog.appendChild(bubble);
    elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

// Handle chat submission
async function handleChatSubmit(event) {
    event.preventDefault();
    const message = elements.chatInput.value.trim();
    if (!message) return;
    appendChatMessage("user", message);
    elements.chatInput.value = "";
    try {
        const response = await fetchJson("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        appendChatMessage("bot", response.response);
    } catch (error) {
        appendChatMessage("bot", "Unable to respond right now.");
    }
}

// Register all event listeners
function registerEventListeners() {
    elements.manualForm.addEventListener("submit", (event) => {
        event.preventDefault();
        submitRecommendation("manual");
    });
    elements.autoForm.addEventListener("submit", (event) => {
        event.preventDefault();
        submitRecommendation("auto");
    });
    elements.manualTab.addEventListener("click", () => toggleMode("manual"));
    elements.autoTab.addEventListener("click", () => toggleMode("auto"));
    elements.manualDistrict.addEventListener("change", (event) => {
        handleDistrictChange(event.target.value, "manual");
    });
    elements.autoDistrict.addEventListener("change", (event) => {
        handleDistrictChange(event.target.value, "auto");
        refreshAutoDefaults();
    });
    elements.autoSeason.addEventListener("change", refreshAutoDefaults);
    document.getElementById("lang-en").addEventListener("click", () => setLanguage("en"));
    document.getElementById("lang-te").addEventListener("click", () => setLanguage("te"));
    elements.chatForm.addEventListener("submit", handleChatSubmit);
}

// Initialize the application
async function init() {
    setLanguage("en");
    registerEventListeners();
    await loadDistricts();
    await loadSchemes();
}

// Start the app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
