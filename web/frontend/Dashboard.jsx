import React, { useState, useEffect } from 'react';
import { Line, Bar, Scatter } from 'react-chartjs-2';
import { useWebSocket } from 'react-use-websocket';

const QuantumDashboard = () => {
  const [quantumMetrics, setQuantumMetrics] = useState(null);
  const [mofFleet, setMofFleet] = useState([]);
  const [rsofcData, setRsofcData] = useState([]);
  
  // WebSocket for live updates
  const { lastMessage } = useWebSocket('ws://localhost:8000/ws/dashboard/live');
  
  useEffect(() => {
    fetch('/api/v1/dashboard/quantum')
      .then(res => res.json())
      .then(data => setQuantumMetrics(data));
      
    fetch('/api/v1/dashboard/mof-fleet')
      .then(res => res.json())
      .then(data => setMofFleet(data));
      
    fetch('/api/v1/dashboard/rsofc')
      .then(res => res.json())
      .then(data => setRsofcData(data));
  }, []);
  
  return (
    <div className="quantum-dashboard">
      <h1>AEQ Quantum Intelligence Console</h1>
      <div className="metrics-grid">
        {/* Quantum metrics card */}
        <div className="metric-card">
          <h3>600+ Qubit Quantum System</h3>
          <p>Quantum Volume: 512</p>
          <p>Error Rate: 0.0032</p>
          <p>VQE Speedup: 47,000x</p>
        </div>
        
        {/* MOF fleet status */}
        <div className="metric-card">
          <h3>MOF Fleet Status</h3>
          {mofFleet.map(site => (
            <div key={site.site}>
              <p>{site.site}: {site.capacity_tons_day} t/day</p>
            </div>
          ))}
        </div>
        
        {/* RSOFC performance */}
        <div className="metric-card">
          <h3>RSOFC Performance</h3>
          {rsofcData.map(site => (
            <div key={site.site}>
              <p>{site.site}: {site.power_mw} MW</p>
              <p>Efficiency: {site.efficiency_pct}%</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QuantumDashboard;
