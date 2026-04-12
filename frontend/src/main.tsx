import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '@/Index';
import '@/index.css'; // Tailwind CSS entry point

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
