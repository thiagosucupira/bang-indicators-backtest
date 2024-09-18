import React, { createContext, useState, useEffect } from 'react';
import axios from 'axios';

export const AuthContext = createContext();

const AuthProvider = ({ children }) => {
    const [auth, setAuth] = useState({
        token: localStorage.getItem('token') || '',
        user: null,
    });

    useEffect(() => {
        if (auth.token) {
            // Decode token to get user info
            try {
                const payload = JSON.parse(atob(auth.token.split('.')[1]));
                setAuth((prev) => ({ ...prev, user: { username: payload.sub, tier: payload.tier } }));
            } catch (e) {
                console.error('Failed to parse token', e);
                setAuth({ token: '', user: null });
            }
        }
    }, [auth.token]);

    const login = async (username, password) => {
        try {
            const formData = new URLSearchParams();
            formData.append('username', username);
            formData.append('password', password);
            
            const response = await axios.post('http://localhost:8000/auth/login', formData.toString(), {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            });
            localStorage.setItem('token', response.data.access_token);
            setAuth({
                token: response.data.access_token,
                user: { username, tier: parseJwt(response.data.access_token).tier },
            });
        } catch (error) {
            console.error('Login error:', error);
            if (error.response) {
                throw new Error(error.response.data.detail || 'Login failed');
            } else if (error.request) {
                throw new Error('No response received from the server');
            } else {
                throw new Error('Error setting up the request');
            }
        }
    };

    const signup = async (username, email, password, tier) => {
        const response = await axios.post('http://localhost:8000/auth/signup', {
            username,
            email,
            password,
            tier,
        });
        return response.data;
    };
    const logout = () => {
        localStorage.removeItem('token');
        setAuth({ token: '', user: null });
    };

    const parseJwt = (token) => {
        try {
            return JSON.parse(atob(token.split('.')[1]));
        } catch (e) {
            return null;
        }
    };

    return (
        <AuthContext.Provider value={{ auth, login, signup, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export default AuthProvider;