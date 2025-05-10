import React from 'react';
import '../styles/Navbar.css';
import logo from '../assets/logo.png'; // Add your logo file to the assets folder

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <img src={logo} alt="Logo" className="navbar-logo" />
        <h1>Shot Analyzer</h1>
      </div>
    </nav>
  );
}

export default Navbar;