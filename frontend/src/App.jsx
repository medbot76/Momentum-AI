"use client"

import React, { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import Register from './pages/Register';
import Login from './pages/Login';
import Wrapper from './pages/Wrapper';
import AuthCallback from './pages/AuthCallback';

function App() {


  return (
    <Router>
      <Routes>
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/auth/callback" element={<AuthCallback />} />
        <Route 
          path="/" 
          element={
            <Wrapper>
              <Home />
            </Wrapper>    
          } 
        />
      </Routes>
    </Router>
  );
}

export default App;