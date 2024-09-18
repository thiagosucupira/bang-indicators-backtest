import React, { useState, useEffect, useContext } from 'react';
import {
    BrowserRouter as Router,
    Routes,
    Route,
    Navigate,
} from 'react-router-dom';
import {
    Container,
    Typography,
    CircularProgress,
    Alert,
    CssBaseline,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Box,
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import axios from 'axios';
import Navbar from './components/Navbar';
import AuthProvider, { AuthContext } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './components/Auth/Login';
import Signup from './components/Auth/Signup';
import IndicatorSelector from './components/IndicatorSelector';
import CustomDatePicker from './components/DatePicker';
import ResultsDisplay from './components/ResultsDisplay';
import GenerateButton from './components/GenerateButton';
import BacktestButton from './components/BacktestButton';

const darkTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#FFD700', // Golden color
        },
        secondary: {
            main: '#f48fb1',
        },
    },
});

const MainApp = () => {
    const { auth } = useContext(AuthContext);
    const [tickers, setTickers] = useState([]);
    const [selectedIndicator, setSelectedIndicator] = useState('FairValueGap');
    const [symbol, setSymbol] = useState('');
    const [interval, setInterval] = useState('1h');
    const [startDate, setStartDate] = useState(new Date(new Date().setDate(1)));
    const [endDate, setEndDate] = useState(new Date());
    const [metrics, setMetrics] = useState(null);
    const [trades, setTrades] = useState(null);
    const [plotImage, setPlotImage] = useState(null);
    const [loadingGenerate, setLoadingGenerate] = useState(false);
    const [loadingBacktest, setLoadingBacktest] = useState(false);
    const [errorGenerate, setErrorGenerate] = useState(null);
    const [errorBacktest, setErrorBacktest] = useState(null);

    useEffect(() => {
        const fetchTickers = async () => {
            try {
                const response = await axios.get('http://localhost:8000/tickers', {
                    headers: {
                        Authorization: `Bearer ${auth.token}`,
                    },
                });
                setTickers(response.data.tickers);
                if (response.data.tickers.length > 0) {
                    setSymbol(response.data.tickers[0].Symbol);
                }
            } catch (err) {
                console.error(err);
                setErrorGenerate('Unable to load tickers.');
            }
        };

        if (auth.token) {
            fetchTickers();
        }
    }, [auth.token]);
    const handleGenerate = async () => {
        setLoadingGenerate(true);
        setErrorGenerate(null);
        setPlotImage(null);
        setMetrics(null);
        setTrades(null);

        try {
            const response = await axios.post('http://localhost:8000/generate_plot', {
                indicator: selectedIndicator,
                symbol: symbol,
                interval: interval,
                start_date: startDate.toISOString().split('T')[0],
                end_date: endDate.toISOString().split('T')[0],
            }, {
                headers: {
                    Authorization: `Bearer ${auth.token}`,
                },
            });

            setPlotImage(`data:image/png;base64,${response.data.plot_image}`);
            setMetrics(response.data.metrics);
            setTrades(response.data.closed_trades);
        } catch (err) {
            setErrorGenerate(err.response?.data.detail || 'Failed to generate plot');
        } finally {
            setLoadingGenerate(false);
        }
    };

    const handleBacktest = async () => {
        setLoadingBacktest(true);
        setErrorBacktest(null);
        setMetrics(null);
        setTrades(null);
        try {
            const response = await axios.post('http://localhost:8000/api/backtest', {
                symbol: symbol,
                interval: interval,
                start_date: startDate.toISOString().split('T')[0],
                end_date: endDate.toISOString().split('T')[0],
            }, {
                headers: {
                    Authorization: `Bearer ${auth.token}`,
                },
            });

            setMetrics(response.data.metrics);
            setTrades(response.data.closed_trades);
        } catch (err) {
            setErrorBacktest(err.response?.data.detail || 'Failed to backtest');
        } finally {
            setLoadingBacktest(false);
        }
    };
    return (
        <Container>
            <Typography variant="h6" gutterBottom>
                Welcome, {auth.user.username} ({auth.user.tier})
            </Typography>
            <Typography variant="h4" gutterBottom style={{ marginTop: '16px' }}>
                BANG Indicators and Backtests
            </Typography>

            <IndicatorSelector
                selectedIndicator={selectedIndicator}
                setSelectedIndicator={setSelectedIndicator}
            />

            <Box sx={{ marginTop: 2 }}>
                <FormControl fullWidth>
                    <InputLabel id="symbol-label">Select Ticker</InputLabel>
                    <Select
                        labelId="symbol-label"
                        id="symbol"
                        value={symbol}
                        label="Select Ticker"
                        onChange={(e) => setSymbol(e.target.value)}
                    >
                        {tickers.map((ticker) => (
                            <MenuItem key={ticker.Symbol} value={ticker.Symbol}>
                                {ticker.Name} ({ticker.Symbol})
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </Box>

            <Box sx={{ marginTop: 2 }}>
                <FormControl fullWidth>
                    <InputLabel id="interval-label">Select Interval</InputLabel>
                    <Select
                        labelId="interval-label"
                        id="interval"
                        value={interval}
                        label="Select Interval"
                        onChange={(e) => setInterval(e.target.value)}
                    >
                        <MenuItem value="1m">1 Minute</MenuItem>
                        <MenuItem value="5m">5 Minutes</MenuItem>
                        <MenuItem value="15m">15 Minutes</MenuItem>
                        <MenuItem value="30m">30 Minutes</MenuItem>
                        <MenuItem value="1h">1 Hour</MenuItem>
                        <MenuItem value="4h">4 Hours</MenuItem>
                        <MenuItem value="1d">1 Day</MenuItem>
                    </Select>
                </FormControl>
            </Box>

            <Box sx={{ marginTop: 2, display: 'flex', gap: 2 }}>
                <CustomDatePicker
                    startDate={startDate}
                    setStartDate={setStartDate}
                    endDate={endDate}
                    setEndDate={setEndDate}
                />
            </Box>

            <Box sx={{ marginTop: 2, display: 'flex', gap: 2 }}>
                <GenerateButton handleGenerate={handleGenerate} disabled={loadingGenerate} />
                {auth.user && auth.user.tier === 'Pro' && (
                    <BacktestButton handleBacktest={handleBacktest} disabled={loadingBacktest} />
                )}
            </Box>

            <Box sx={{ marginTop: 2 }}>
                {loadingGenerate && <CircularProgress />}
                {loadingBacktest && <CircularProgress />}
            </Box>

            <Box sx={{ marginTop: 2 }}>
                {errorGenerate && (
                    <Alert severity="error">{errorGenerate}</Alert>
                )}
                {errorBacktest && (
                    <Alert severity="error">{errorBacktest}</Alert>
                )}
            </Box>

            {plotImage && (
                <Box sx={{ marginTop: 4, textAlign: 'center' }}>
                    <Typography variant="h5" gutterBottom>
                        Fair Value Gaps Plot
                    </Typography>
                    <img src={plotImage} alt="Fair Value Gaps" style={{ maxWidth: '100%' }} />
                </Box>
            )}

            {metrics && trades && (
                <ResultsDisplay metrics={metrics} trades={trades} />
            )}
        </Container>
    );
};

const App = () => {
    return (
        <AuthProvider>
            <ThemeProvider theme={darkTheme}>
                <CssBaseline />
                <Router>
                    <Navbar />
                    <Routes>
                        <Route path="/login" element={<Login />} />
                        <Route path="/signup" element={<Signup />} />
                        <Route path="/" element={<Navigate to="/login" />} />
                        <Route element={<ProtectedRoute />}>
                            <Route path="/main" element={<MainApp />} /> 
                        </Route>
                    </Routes>
                </Router>
            </ThemeProvider>
        </AuthProvider>
    );
};

export default App;