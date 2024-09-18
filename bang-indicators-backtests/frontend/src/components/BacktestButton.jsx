import React from 'react';
import { Button } from '@mui/material';

const BacktestButton = ({ handleBacktest, disabled }) => {
    return (
        <Button
            variant="contained"
            onClick={handleBacktest}
            disabled={disabled}
            fullWidth
            sx={{
                background: 'linear-gradient(45deg, #FFD700 30%, #FFC107 90%)',
                color: '#000',
                borderRadius: '12px',
                boxShadow: '0 3px 5px 2px rgba(255, 215, 0, .3)',
                fontWeight: 'bold',
                height: '50px',
                '&:hover': {
                    background: 'linear-gradient(45deg, #FFC107 30%, #FFD700 90%)',
                },
                transition: 'background 0.3s ease',
            }}
        >
            Backtest
        </Button>
    );
};

export default BacktestButton;