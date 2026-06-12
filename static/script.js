// script.js
document.addEventListener("DOMContentLoaded", function () {
    
    // Global variable to store loaded metrics
    let bestModelName = "";
    
    // ─────────────────────────────────────────────────────────────────────────
    // 1. PAGE ROUTING & NAVIGATION
    // ─────────────────────────────────────────────────────────────────────────
    const navButtons = document.querySelectorAll(".nav-btn");
    const sections = document.querySelectorAll(".view-section");
    
    navButtons.forEach(btn => {
        btn.addEventListener("click", function () {
            // Remove active class from all buttons
            navButtons.forEach(b => b.classList.remove("active"));
            // Add active class to clicked button
            this.classList.add("active");
            
            // Hide all sections
            sections.forEach(sec => sec.classList.remove("active"));
            
            // Show targeted section
            const targetId = this.getAttribute("data-target");
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.add("active");
            }
        });
    });

    // ─────────────────────────────────────────────────────────────────────────
    // 2. SLIDER DISPLAY VALUE CONTROLLERS
    // ─────────────────────────────────────────────────────────────────────────
    const sliders = document.querySelectorAll(".slider");
    sliders.forEach(slider => {
        const display = document.getElementById(slider.id + "_val");
        if (display) {
            slider.addEventListener("input", function () {
                display.textContent = parseFloat(this.value).toFixed(slider.step % 1 === 0 ? 0 : 1);
            });
        }
    });



    // ─────────────────────────────────────────────────────────────────────────
    // 3. API DATA INITIALIZATION ON STARTUP
    // ─────────────────────────────────────────────────────────────────────────
    
    // 3.1 Fetch Metrics
    fetch("/api/metrics")
        .then(res => res.json())
        .then(data => {
            const metrics = data.metrics;
            const models = Object.keys(metrics.MAE);

            // Find the best model based on Method 1 (100 - MAPE)
            let bestAccuracy = -1;
            let bestModelByAcc = "";
            models.forEach(model => {
                const mapeVal = metrics["MAPE (%)"][model];
                const accVal = 100 - mapeVal;
                if (accVal > bestAccuracy) {
                    bestAccuracy = accVal;
                    bestModelByAcc = model;
                }
            });
            bestModelName = bestModelByAcc; // Set this as our best model name

            // Find R-squared key dynamically (e.g. "R²", "R2", etc.)
            const r2Key = Object.keys(metrics).find(k => k.startsWith("R") && k !== "RMSE") || "R²";

            // Populate Overview Page showcases
            document.getElementById("best-model-name-title").textContent = "🏆 " + bestModelName;
            
            const bestMAE = metrics.MAE[bestModelName];
            const bestRMSE = metrics.RMSE[bestModelName];
            
            document.getElementById("sidebar-r2-val").textContent = bestAccuracy.toFixed(2) + "%";
            document.getElementById("overview-r2").textContent = bestAccuracy.toFixed(2) + "%";
            document.getElementById("overview-mae").textContent = bestMAE.toFixed(4);
            document.getElementById("overview-rmse").textContent = bestRMSE.toFixed(4);

            // Populate Model Performance Page Callout
            document.getElementById("performance-best-model-text").textContent = 
                `Selected Production Model: ${bestModelName} (Accuracy = ${bestAccuracy.toFixed(2)}%)`;

            // Update flowchart dynamically based on bestModelName
            const arrow = document.getElementById("flowchart-success-arrow");
            const captionName = document.getElementById("flowchart-best-model-caption-name");
            if (captionName) {
                captionName.textContent = bestModelName;
            }
            
            const models_ids = {
                'Linear Regression': { rect: 'flowchart-rect-lr', tag: 'flowchart-tag-lr', text: 'flowchart-text-lr', y_start: 45 },
                'Random Forest': { rect: 'flowchart-rect-rf', tag: 'flowchart-tag-rf', text: 'flowchart-text-rf', y_start: 110 },
                'Gradient Boosting': { rect: 'flowchart-rect-gb', tag: 'flowchart-tag-gb', text: 'flowchart-text-gb', y_start: 175 }
            };

            Object.keys(models_ids).forEach(name => {
                const info = models_ids[name];
                const rectEl = document.getElementById(info.rect);
                const tagEl = document.getElementById(info.tag);
                const textEl = document.getElementById(info.text);
                
                if (rectEl && tagEl && textEl) {
                    if (name === bestModelName) {
                        rectEl.classList.add("active-node");
                        textEl.classList.add("bold");
                        tagEl.style.display = "block";
                        if (arrow) {
                            arrow.setAttribute("d", `M 440 ${info.y_start} L 570 110`);
                        }
                    } else {
                        rectEl.classList.remove("active-node");
                        textEl.classList.remove("bold");
                        tagEl.style.display = "none";
                    }
                }
            });

            // Populate Metrics Comparison Table
            const tbody = document.getElementById("metrics-table-body");
            tbody.innerHTML = ""; // clear loading text
            
            models.forEach(model => {
                const row = document.createElement("tr");
                if (model === bestModelName) {
                    row.classList.add("metrics-highlight-row");
                }
                
                const r2 = metrics[r2Key] ? metrics[r2Key][model] : 0;
                const mae = metrics.MAE[model];
                const rmse = metrics.RMSE[model];
                const mape = metrics["MAPE (%)"][model];
                const accuracy = 100 - mape;
                
                row.innerHTML = `
                    <td><strong>${model}</strong>${model === bestModelName ? ' 🌟' : ''}</td>
                    <td><strong>${accuracy.toFixed(2)}%</strong></td>
                    <td>${mae.toFixed(4)}</td>
                    <td>${rmse.toFixed(4)}</td>
                    <td>${r2.toFixed(4)}</td>
                `;
                tbody.appendChild(row);
            });

            // Populate EDA Summary Table
            // Using test results statistics as standard proxy
            document.getElementById("stat-count").textContent = "3491.0000";
            document.getElementById("stat-mean").textContent = "0.9637";
            document.getElementById("stat-std").textContent = "0.8929";
            document.getElementById("stat-min").textContent = "0.1240";
            document.getElementById("stat-25").textContent = "0.2867";
            document.getElementById("stat-median").textContent = "0.5519";
            document.getElementById("stat-75").textContent = "1.4578";
            document.getElementById("stat-max").textContent = "6.5605";

            // Render Performance Page Math Formulas
            try {
                katex.render(
                    "\\text{Accuracy (\\%)} = 100\\% - \\text{MAPE (\\%)} = 100\\% - \\left( \\frac{100\\%}{n} \\sum_{i=1}^{n} \\left| \\frac{y_i - \\hat{y}_i}{y_i} \\right| \\right)",
                    document.getElementById("latex-accuracy-formula"),
                    { throwOnError: false, displayMode: true }
                );
                katex.render(
                    "R^2 = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}",
                    document.getElementById("latex-r2-formula"),
                    { throwOnError: false, displayMode: true }
                );
                katex.render(
                    "\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|",
                    document.getElementById("latex-mae-formula"),
                    { throwOnError: false, displayMode: true }
                );
            } catch (err) {
                console.error("Error rendering KaTeX on performance page:", err);
            }
        })
        .catch(err => console.error("Error loading metrics:", err));


    // ─────────────────────────────────────────────────────────────────────────
    // 4. SUBMIT PREDICTION HANDLER
    // ─────────────────────────────────────────────────────────────────────────
    const form = document.getElementById("prediction-form");
    const resultPanel = document.getElementById("prediction-result-panel");

    form.addEventListener("submit", function (e) {
        e.preventDefault();
        
        // Collect form data
        const formData = new FormData(form);
        const inputs = {
            temperature: parseFloat(formData.get("temperature")),
            hour: parseInt(formData.get("hour")),
            season: parseInt(formData.get("season")),
            voltage: parseFloat(formData.get("voltage")),
            lag_1: parseFloat(formData.get("lag_1")),
            lag_24: parseFloat(formData.get("lag_24")),
            current_intensity: parseFloat(formData.get("current_intensity"))
        };

        // POST request to prediction endpoint
        fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(inputs)
        })
        .then(res => res.json())
        .then(data => {
            const prediction = data.prediction;
            const basePreds = data.base_predictions;
            
            // Show result panel
            resultPanel.classList.add("show");
            
            // Scroll to results smoothly
            resultPanel.scrollIntoView({ behavior: 'smooth' });

            // Set main prediction values
            document.getElementById("result-predicted-kw").textContent = `${prediction.toFixed(4)} kW`;
            document.getElementById("result-predicted-kw-sub").textContent = `${prediction.toFixed(4)} kW`;

            // Set base model predictions
            document.getElementById("pred-lr").textContent = `${basePreds["Linear Regression"].toFixed(4)} kW`;
            document.getElementById("pred-rf").textContent = `${basePreds["Random Forest"].toFixed(4)} kW`;
            document.getElementById("pred-gb").textContent = `${basePreds["Gradient Boosting"].toFixed(4)} kW`;

            // Style consumption level card
            const levelCard = document.getElementById("result-level-card");
            const levelIcon = document.getElementById("result-level-icon");
            const levelText = document.getElementById("result-level-text");
            
            let level = "";
            let color = "";
            let icon = "";
            
            if (prediction < 1.0) {
                level = "Very Low"; color = "#27AE60"; icon = "🟢";
            } else if (prediction < 1.5) {
                level = "Low";      color = "#2ECC71"; icon = "🟢";
            } else if (prediction < 2.0) {
                level = "Moderate"; color = "#F39C12"; icon = "🟡";
            } else if (prediction < 2.5) {
                level = "High";     color = "#E67E22"; icon = "🟠";
            } else {
                level = "Very High";color = "#E74C3C"; icon = "🔴";
            }

            levelText.textContent = level;
            levelText.style.color = color;
            levelIcon.textContent = icon;
            levelCard.style.borderLeft = `6px solid ${color}`;


        })
        .catch(err => {
            console.error("Error executing prediction:", err);
            alert("An error occurred during model prediction execution.");
        });
    });

});
