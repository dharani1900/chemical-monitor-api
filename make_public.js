const localtunnel = require('localtunnel');
const fs = require('fs');
const path = require('path');

(async () => {
  try {
    const tunnel = await localtunnel({ port: 5173 });
    const output = `PUBLIC_URL: ${tunnel.url}`;
    console.log(output);
    fs.writeFileSync(path.join(__dirname, 'public_url.txt'), output);
    tunnel.on('close', () => {
      console.log('Tunnel closed');
    });
  } catch (err) {
    console.error('Tunnel error:', err);
    fs.writeFileSync(path.join(__dirname, 'public_url.txt'), `ERROR: ${err.message}`);
  }
})();
