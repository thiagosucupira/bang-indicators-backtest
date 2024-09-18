import React from 'react';
import { FormControl, InputLabel, Select, MenuItem, Box } from '@mui/material';

const IndicatorSelector = ({ selectedIndicator, setSelectedIndicator }) => {
    return (
        <Box sx={{ marginTop: 2 }}>
            <FormControl fullWidth>
                <InputLabel id="indicator-label">Select Indicator</InputLabel>
                <Select
                    labelId="indicator-label"
                    id="indicator"
                    value={selectedIndicator}
                    label="Select Indicator"
                    onChange={(e) => setSelectedIndicator(e.target.value)}
                >
                    <MenuItem value="FairValueGap">Fair Value Gap</MenuItem>
                    {/* Add more indicators as needed */}
                </Select>
            </FormControl>
        </Box>
    );
};

export default IndicatorSelector;