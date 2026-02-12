# Kai Platform Frontend

Modern React frontend for the Kai PPC Marketing Intelligence Platform.

## Tech Stack

- **React 18** - UI framework
- **Material-UI (MUI)** - Component library
- **Framer Motion** - Animations
- **Vite** - Build tool
- **Axios** - API client
- **React Router** - Routing

## Development

```bash
# Install dependencies
npm install

# Start dev server (with API proxy)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable components
│   │   └── Layout.jsx  # Main layout with navigation
│   ├── pages/          # Page components
│   │   ├── Home.jsx
│   │   ├── KlauditAudit.jsx
│   │   ├── CreativeStudio.jsx
│   │   ├── PMaxDeepDive.jsx
│   │   ├── SerpMonitor.jsx
│   │   └── Settings.jsx
│   ├── App.jsx         # Main app component
│   ├── main.jsx        # Entry point
│   └── index.css       # Global styles
├── public/             # Static assets
├── index.html          # HTML template
├── vite.config.js      # Vite configuration
└── package.json        # Dependencies
```

## API Integration

The frontend connects to the FastAPI backend at `/api/*`. In development, Vite proxies these requests to `http://localhost:8000`.

## Deployment

Build the frontend and deploy to Azure Static Web Apps:

```bash
npm run build
# Deploy the 'dist' folder to Azure Static Web Apps
```
