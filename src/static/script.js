document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const priceValue = document.getElementById('price-value');
    const submitBtn = document.getElementById('submit-btn');

    // Load configuration (neighbourhoods, room types)
    fetch('/config')
        .then(response => response.json())
        .then(data => {
            const nbSelect = document.getElementById('neighbourhood');
            const rtSelect = document.getElementById('room_type');

            if (data.neighbourhoods) {
                nbSelect.innerHTML = '<option value="">Select Neighbourhood</option>';
                data.neighbourhoods.forEach(nb => {
                    const option = document.createElement('option');
                    option.value = nb;
                    option.textContent = nb;
                    nbSelect.appendChild(option);
                });
            }

            if (data.room_types) {
                rtSelect.innerHTML = '<option value="">Select Room Type</option>';
                data.room_types.forEach(rt => {
                    const option = document.createElement('option');
                    option.value = rt;
                    option.textContent = rt;
                    rtSelect.appendChild(option);
                });
            }
        })
        .catch(err => console.error('Error loading config:', err));

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = 'Calculating...';
        resultDiv.classList.add('hidden');

        // Gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Handle amenities (multi-select checkbox)
        const amenities = formData.getAll('amenities');
        data.amenities = amenities;

        // Handle boolean/checkbox for superhost
        data.host_is_superhost = form.querySelector('#host_is_superhost').checked;

        // Ensure numbers are numbers
        ['latitude', 'longitude', 'accommodates', 'bedrooms', 'beds'].forEach(field => {
            if (data[field]) data[field] = parseFloat(data[field]);
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (response.ok) {
                priceValue.textContent = `$${result.predicted_price.toFixed(2)}`;
                resultDiv.classList.remove('hidden');
                // Scroll to result
                resultDiv.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert(`Error: ${result.error || 'Unknown error occurred'}`);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to get prediction. Check console for details.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Predict Price';
        }
    });
});

