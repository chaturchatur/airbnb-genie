document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const priceValue = document.getElementById('price-value');
    const submitBtn = document.getElementById('submit-btn');

    // store neighbourhood coordinates from config
    let neighbourhoodCoords = {};

    // preset definitions for demo
    const presets = {
        solo: {
            name: 'Cozy Studio for Solo Traveler',
            neighbourhood: 'Shaw, Logan Circle',
            room_type: 'Private room',
            accommodates: 1,
            bedrooms: 1,
            beds: 1,
            bathrooms_text: '1 shared bath',
            amenities: ['Wifi', 'Heating'],
            host_response_time: 'within an hour',
            host_is_superhost: false
        },
        couple: {
            name: 'Romantic Getaway in Dupont Circle',
            neighbourhood: 'Dupont Circle, Connecticut Avenue/K Street',
            room_type: 'Entire home/apt',
            accommodates: 2,
            bedrooms: 1,
            beds: 1,
            bathrooms_text: '1 bath',
            amenities: ['Wifi', 'Kitchen', 'Air conditioning', 'Heating'],
            host_response_time: 'within an hour',
            host_is_superhost: true
        },
        family: {
            name: 'Spacious Family Home in Georgetown',
            neighbourhood: 'Georgetown, Burleith/Hillandale',
            room_type: 'Entire home/apt',
            accommodates: 4,
            bedrooms: 2,
            beds: 3,
            bathrooms_text: '2 baths',
            amenities: ['Wifi', 'Kitchen', 'Air conditioning', 'Washer', 'Dryer', 'Heating'],
            host_response_time: 'within an hour',
            host_is_superhost: true
        },
        group: {
            name: 'Large Group House near Capitol Hill',
            neighbourhood: 'Capitol Hill, Lincoln Park',
            room_type: 'Entire home/apt',
            accommodates: 6,
            bedrooms: 3,
            beds: 5,
            bathrooms_text: '2.5 baths',
            amenities: ['Wifi', 'Kitchen', 'Air conditioning', 'Washer', 'Dryer', 'Heating'],
            host_response_time: 'within a few hours',
            host_is_superhost: true
        }
    };

    // load configuration (neighbourhoods, room types, coords)
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

            // store neighbourhood coordinates
            if (data.neighbourhood_coords) {
                neighbourhoodCoords = data.neighbourhood_coords;
            }
        })
        .catch(err => console.error('error loading config:', err));

    // auto-fill coordinates when neighbourhood changes
    document.getElementById('neighbourhood').addEventListener('change', (e) => {
        const selected = e.target.value;
        if (selected && neighbourhoodCoords[selected]) {
            document.getElementById('latitude').value = neighbourhoodCoords[selected].lat;
            document.getElementById('longitude').value = neighbourhoodCoords[selected].lng;
        }
    });

    // apply preset to form
    function applyPreset(presetKey) {
        const preset = presets[presetKey];
        if (!preset) return;

        // fill text fields
        document.getElementById('name').value = preset.name;
        document.getElementById('neighbourhood').value = preset.neighbourhood;
        document.getElementById('room_type').value = preset.room_type;
        document.getElementById('accommodates').value = preset.accommodates;
        document.getElementById('bedrooms').value = preset.bedrooms;
        document.getElementById('beds').value = preset.beds;
        document.getElementById('bathrooms_text').value = preset.bathrooms_text;
        document.getElementById('host_response_time').value = preset.host_response_time;
        document.getElementById('host_is_superhost').checked = preset.host_is_superhost;

        // fill coordinates based on neighbourhood
        if (neighbourhoodCoords[preset.neighbourhood]) {
            document.getElementById('latitude').value = neighbourhoodCoords[preset.neighbourhood].lat;
            document.getElementById('longitude').value = neighbourhoodCoords[preset.neighbourhood].lng;
        }

        // handle amenities checkboxes
        const amenityCheckboxes = document.querySelectorAll('input[name="amenities"]');
        amenityCheckboxes.forEach(cb => {
            cb.checked = preset.amenities.includes(cb.value);
        });

        // highlight active preset card
        document.querySelectorAll('.preset-card').forEach(card => {
            card.classList.remove('active');
        });
        document.querySelector(`[data-preset="${presetKey}"]`).classList.add('active');

        // auto-submit the form
        form.dispatchEvent(new Event('submit'));
    }

    // preset card click handlers
    document.querySelectorAll('.preset-card').forEach(card => {
        card.addEventListener('click', () => {
            const presetKey = card.dataset.preset;
            applyPreset(presetKey);
        });
    });

    // form submission handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = 'Calculating...';
        resultDiv.classList.add('hidden');

        // gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // handle amenities (multi-select checkbox)
        const amenities = formData.getAll('amenities');
        data.amenities = amenities;

        // handle boolean/checkbox for superhost
        data.host_is_superhost = form.querySelector('#host_is_superhost').checked;

        // ensure numbers are numbers
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
                // scroll to result
                resultDiv.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert(`error: ${result.error || 'unknown error occurred'}`);
            }
        } catch (error) {
            console.error('error:', error);
            alert('failed to get prediction - check console for details');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Predict Price';
        }
    });
});
