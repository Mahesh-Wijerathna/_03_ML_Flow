const express = require('express');
const axios = require('axios');
const path = require('path');
const app = express();

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public'))); // Serve static files

app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post('http://backend:5001/predict', req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Prediction failed' });
    }
});

app.listen(3000, () => {
    console.log('Frontend server running on port 3000');
});