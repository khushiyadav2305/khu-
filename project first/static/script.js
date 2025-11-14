// // // // // // script.js - JavaScript for interactivity in the share price prediction project

// // // // // async function predictStock(event) {
// // // // //     event.preventDefault();
// // // // //     const ticker = document.getElementById('ticker').value;
    
// // // // //     try {
// // // // //         const response = await fetch('/predict', {
// // // // //             method: 'POST',
// // // // //             headers: { 'Content-Type': 'application/json' },
// // // // //             body: JSON.stringify({ ticker })
// // // // //         });
        
// // // // //         if (!response.ok) {
// // // // //             throw new Error('Network response was not ok');
// // // // //         }
        
// // // // //         const data = await response.json();
        
// // // // //         if (data.error) {
// // // // //             document.getElementById('result').innerHTML = `<p>Error: ${data.error}</p>`;
// // // // //             return;
// // // // //         }
        
// // // // //         // Display historical data
// // // // //         let historicalHtml = '<h3>Recent Historical Closes:</h3><ul>';
// // // // //         for (const [date, price] of Object.entries(data.historical)) {
// // // // //             historicalHtml += `<li>${date}: $${price.toFixed(2)}</li>`;
// // // // //         }
// // // // //         historicalHtml += '</ul>';
// // // // //         document.getElementById('historical').innerHTML = historicalHtml;
        
// // // // //         // Display predictions
// // // // //         let predictionsHtml = '<h3>Future Predictions (Next 30 Days):</h3><ul>';
// // // // //         data.predictions.forEach((price, index) => {
// // // // //             predictionsHtml += `<li>Day ${index + 1}: $${price.toFixed(2)}</li>`;
// // // // //         });
// // // // //         predictionsHtml += '</ul>';
// // // // //         document.getElementById('predictions').innerHTML = predictionsHtml;
        
// // // // //         // Display plot
// // // // //         document.getElementById('plot').src = `data:image/png;base64,${data.plot}`;
        
// // // // //     } catch (error) {
// // // // //         console.error('Error:', error);
// // // // //         document.getElementById('result').innerHTML = '<p>An error occurred. Please try again.</p>';
// // // // //     }
// // // // // }



// // // // async function predictStock(event) {
// // // //     event.preventDefault();
// // // //     const ticker = document.getElementById('ticker').value.trim().toUpperCase();
// // // //     const resultDiv = document.getElementById('result');
// // // //     const historicalDiv = document.getElementById('historical');
// // // //     const predictionsDiv = document.getElementById('predictions');
// // // //     const plotImg = document.getElementById('plot');

// // // //     try {
// // // //         const response = await fetch('http://localhost:5000/predict', {
// // // //             method: 'POST',
// // // //             headers: {
// // // //                 'Content-Type': 'application/json',
// // // //             },
// // // //             body: JSON.stringify({ ticker })
// // // //         });

// // // //         const data = await response.json();
// // // //         if (response.ok) {
// // // //             historicalDiv.innerHTML = `<h3>Historical Prices (Last 90 Days):</h3><p>${data.historical.map(price => price.toFixed(2)).join(', ')}</p>`;
// // // //             predictionsDiv.innerHTML = `<h3>Predicted Prices (Next 30 Days):</h3><p>${data.predictions.map(price => price.toFixed(2)).join(', ')}</p>`;
// // // //             plotImg.src = data.plot;
// // // //         } else {
// // // //             resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
// // // //         }
// // // //     } catch (error) {
// // // //         resultDiv.innerHTML = `<p style="color: red;">Error: Could not connect to the server</p>`;
// // // //         console.error('Error:', error);
// // // //     }
// // // // }






// // // //.........................................2nd try............................................................






// // // async function predictStock(event) {
// // //     event.preventDefault();
// // //     const ticker = document.getElementById('ticker').value.trim().toUpperCase();
// // //     const resultDiv = document.getElementById('result');
// // //     const historicalDiv = document.getElementById('historical');
// // //     const predictionsDiv = document.getElementById('predictions');
// // //     const plotImg = document.getElementById('plot');

// // //     resultDiv.innerHTML = '<p>Loading...</p>';

// // //     try {
// // //         // Fetch current price
// // //         const priceResponse = await fetch('http://localhost:5000/current_price', {
// // //             method: 'POST',
// // //             headers: { 'Content-Type': 'application/json' },
// // //             body: JSON.stringify({ ticker })
// // //         });
// // //         const priceData = await priceResponse.json();

// // //         // Fetch predictions
// // //         const predictResponse = await fetch('http://localhost:5000/predict', {
// // //             method: 'POST',
// // //             headers: { 'Content-Type': 'application/json' },
// // //             body: JSON.stringify({ ticker })
// // //         });
// // //         const predictData = await predictResponse.json();

// // //         // Display results
// // //         if (priceResponse.ok && predictResponse.ok) {
// // //             resultDiv.innerHTML = `<h3>Current Price for ${ticker}: $${priceData.current_price.toFixed(2)}</h3>`;
// // //             historicalDiv.innerHTML = `<h3>Historical Prices (Last 90 Days):</h3><p>${predictData.historical.map(price => price.toFixed(2)).join(', ')}</p>`;
// // //             predictionsDiv.innerHTML = `<h3>Predicted Prices (Next 30 Days):</h3><p>${predictData.predictions.map(price => price.toFixed(2)).join(', ')}</p>`;
// // //             plotImg.src = predictData.plot;
// // //         } else {
// // //             resultDiv.innerHTML = `<p style="color: red;">Error: ${priceData.error || predictData.error}</p>`;
// // //         }
// // //     } catch (error) {
// // //         resultDiv.innerHTML = `<p style="color: red;">Error: Could not connect to the server</p>`;
// // //         console.error('Error:', error);
// // //     }
// // // }
// // // ....................................................................................................................................


// // function predict() {
// //     const ticker = document.getElementById('ticker').value;
// //     const start = document.getElementById('start').value;
// //     const end = document.getElementById('end').value;
    
// //     if (!start || !end) {
// //         alert('Please enter both start and end dates.');
// //         return;
// //     }
    
// //     fetch('/predict', {
// //         method: 'POST',
// //         headers: { 'Content-Type': 'application/json' },
// //         body: JSON.stringify({ ticker, start, end })
// //     })
// //     .then(response => response.json())
// //     .then(data => {
// //         const resultDiv = document.getElementById('result');
// //         if (data.error) {
// //             resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
// //         } else {
// //             resultDiv.innerHTML = `
// //                 <p><strong>Predicted Next Day Price (INR):</strong> ₹${data.predicted_price_inr.toLocaleString()}</p>
// //                 <p><strong>Model Accuracy (R²):</strong> ${data.accuracy}</p>
// //                 <p><strong>Accuracy Level:</strong> ${data.accuracy_level}</p>
// //             `;
// //         }
// //     })
// //     .catch(error => {
// //         document.getElementById('result').innerHTML = `<p style="color: red;">Request failed: ${error}</p>`;
// //     });
// // }


// const statusEl = document.getElementById('status');
// const resultsTable = document.getElementById('resultsTable');
// const tbody = resultsTable.querySelector('tbody');
// let fgiChart;

// function setStatus(msg, loading=false) {
//   statusEl.innerHTML = loading ? `<span class="spin"></span>${msg}` : msg;
// }

// async function callPredict() {
//   const tickers = document.getElementById('tickers').value.split(',')
//     .map(s => s.trim()).filter(Boolean);
//   const from = document.getElementById('from').value || null;
//   const to = document.getElementById('to').value || null;
//   const horizon = document.getElementById('horizon').value;

//   if (!tickers.length) {
//     alert('Please enter at least one ticker');
//     return;
//   }

//   setStatus('Predicting… this may take ~10–30s depending on tickers.', true);
//   resultsTable.style.display = 'none';
//   tbody.innerHTML = '';

//   try {
//     const res = await fetch('/predict', {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify({ tickers, start: from, end: to, horizon: parseInt(horizon) })
//     });

//     if (!res.ok) throw new Error(`Server error: ${res.status}`);

//     const data = await res.json();
//     if (!Array.isArray(data)) throw new Error(data.error || 'Unexpected response');

//     for (const row of data) {
//       const tr = document.createElement('tr');
//       function td(v){ const el=document.createElement('td'); el.textContent=(v ?? ''); return el; }
//       tr.appendChild(td(row.Ticker));
//       tr.appendChild(td(row.Currency));
//       tr.appendChild(td(row.From));
//       tr.appendChild(td(row.To));
//       tr.appendChild(td(row.Horizon_Days));
//       tr.appendChild(td(row.Last_Close_Local));
//       tr.appendChild(td(row.Last_Close_INR));
//       tr.appendChild(td(row.Predicted_Next_Close_Local));
//       tr.appendChild(td(row.Predicted_Next_Close_INR));
//       tr.appendChild(td(row.MAPE));
//       tr.appendChild(td(row.Accuracy));
//       tr.appendChild(td(row.Note));
//       tbody.appendChild(tr);
//     }
//     resultsTable.style.display = '';
//     setStatus('Done.');
//   } catch (err) {
//     console.error(err);
//     setStatus('Error: ' + err.message);
//   }
// }

// async function loadFgi() {
//   const from = document.getElementById('from').value || '';
//   const to = document.getElementById('to').value || '';

//   setStatus('Loading FGI…', true);
//   try {
//     const params = new URLSearchParams();
//     if (from) params.set('start', from);
//     if (to) params.set('end', to);

//     const res = await fetch('/api/fgi?' + params.toString());
//     if (!res.ok) throw new Error(`Server error: ${res.status}`);

//     const data = await res.json();
//     if (!Array.isArray(data)) throw new Error(data.error || 'Failed to load FGI');

//     const labels = data.map(d => d.date);
//     const values = data.map(d => d.FGI);

//     if (fgiChart) fgiChart.destroy();
//     const ctx = document.getElementById('fgiChart').getContext('2d');
//     fgiChart = new Chart(ctx, {
//       type: 'line',
//       data: { labels, datasets: [{ label: 'Fear & Greed Index', data: values, fill: false }] },
//       options: { 
//         plugins: { legend: { labels: { color:'#cbd5e1' } } }, 
//         scales: { 
//           x: { ticks:{ color:'#a5b3d8' } }, 
//           y: { ticks:{ color:'#a5b3d8' }, suggestedMin:0, suggestedMax:100 } 
//         } 
//       }
//     });
//     setStatus('FGI loaded.');
//   } catch (err) {
//     console.error(err);
//     setStatus('Error: ' + err.message);
//   }
// }

// document.getElementById('predictBtn').addEventListener('click', callPredict);
// document.getElementById('loadFgiBtn').addEventListener('click', loadFgi);
