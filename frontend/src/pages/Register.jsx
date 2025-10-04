import React, { useState } from 'react'
import supabase from '../helper/supabaseClient';
import { Link } from 'react-router-dom';

// Camera icon from Home.jsx
const Camera = ({ className = "w-8 h-8" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

function Register() {

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [message, setMessage] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage('');

        // Check if passwords match before proceeding with registration
        if (password !== confirmPassword) {
            setMessage('Passwords do not match');
            return;
        }

        const { data, error } = await supabase.auth.signUp({
            email,
            password,
        });
        if (error) {
            setMessage(error.message);
            return;
        } 
        if (data) {
            setMessage('Account created successfully');
        }
        
        setEmail('');
        setPassword('');
        setConfirmPassword('');
    }
        
        
    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-50">
            <div className="w-full max-w-sm bg-white rounded-xl shadow p-8 flex flex-col items-center border border-gray-200">
                <div className="flex items-center justify-center w-14 h-14 bg-blue-600 text-white rounded-lg mb-4">
                    <Camera className="w-8 h-8" />
                </div>
                <h1 className="text-2xl font-semibold text-gray-900 mb-2">Create Account</h1>
                <p className="text-sm text-gray-500 mb-6">Join us! Please create your account.</p>
                <br></br>
                {message && <span className="text-sm text-red-600 mb-4">{message}</span>}
                <form onSubmit={handleSubmit} className="w-full flex flex-col gap-4">
                    <input 
                        type="email" 
                        placeholder="Email" 
                        value={email} 
                        onChange={(e) => setEmail(e.target.value)} 
                        required
                        className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <input 
                        type="password" 
                        placeholder="Password" 
                        value={password} 
                        onChange={(e) => setPassword(e.target.value)} 
                        required
                        className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <input 
                        type="password" 
                        placeholder="Confirm Password" 
                        value={confirmPassword} 
                        onChange={(e) => setConfirmPassword(e.target.value)} 
                        required
                        className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button 
                        type="submit"
                        className="mt-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition-colors"
                    >
                        Register
                    </button>
                    <span className="text-sm text-gray-600">Already have an account?</span>
                    <Link to="/login" className="text-blue-600 hover:text-blue-700 text-sm font-medium">Sign In</Link>
                </form>
            </div>
        </div>
    )
}

export default Register