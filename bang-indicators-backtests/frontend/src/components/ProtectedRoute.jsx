import React, { useContext } from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';

const ProtectedRoute = ({ proOnly }) => {
    const { auth } = useContext(AuthContext);

    if (!auth.token) {
        return <Navigate to="/login" replace />;
    }

    if (proOnly && auth.user.tier !== 'Pro') {
        return <Navigate to="/" replace />;
    }

    return <Outlet />;
};

export default ProtectedRoute;