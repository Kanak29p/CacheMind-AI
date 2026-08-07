const app = require('./app');
const config = require('./config/config');

const PORT = config.port;

const server = app.listen(PORT, () => {
  console.log(`=========================================`);
  console.log(` CacheMind AI Proxy Server Starting...`);
  console.log(` Port:        ${PORT}`);
  console.log(` Environment: ${config.nodeEnv}`);
  console.log(`=========================================`);
});

// Handle unhandled promise rejections anywhere in the application (fail-safe)
process.on('unhandledRejection', (err) => {
  console.error(`Unhandled Rejection Error: ${err.message}`);
  // Gracefully shut down the server to finish outstanding requests, then exit
  server.close(() => {
    process.exit(1);
  });
});
