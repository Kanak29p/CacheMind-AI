const express = require('express');

const app = express();

// Global Middleware: Automatically parses incoming JSON request bodies
// and attaches the parsed data to req.body.
app.use(express.json());

// Basic health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'UP',
    timestamp: new Date().toISOString(),
  });
});

module.exports = app;
