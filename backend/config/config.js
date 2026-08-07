const dotenv = require('dotenv');
const path = require('path');

// Load environment variables from the .env file in the backend directory
dotenv.config({ path: path.join(__dirname, '../.env') });

// Define which variables are mandatory for the application to function
const requiredEnvVars = ['OPENAI_API_KEY'];

// Validate that required variables are defined
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`CRITICAL CONFIG ERROR: Missing required environment variable: ${envVar}. Please check your .env file.`);
  }
}

module.exports = {
  port: parseInt(process.env.PORT, 10) || 5000,
  nodeEnv: process.env.NODE_ENV || 'development',
  openaiApiKey: process.env.OPENAI_API_KEY,
};
