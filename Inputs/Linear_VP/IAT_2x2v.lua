-- Gkyl ------------------------------------------------------------------------
local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell()

-- electron parameters
vDriftElc = {0.01,0}   ---modified from 0.159
vtElc = 0.02
-- ion parameters
vDriftIon = {0.0,0.0}
vtIon = 0.000566      ---- modified from 0.001 (use Te = 50 Ti)
-- mass ratio
massRatio = 25.0  ----modified from 25 

knumberx = 10.0 -- wave-number
kunmbery = 0.0
perturbation = 1.0e-4 -- distribution function perturbation

local function maxwellian2v(v, vDrift, vt)
    return 1/(2*math.pi*vt^2)*math.exp(-((v[1]-vDrift[1])^2+(v[2]-vDrift[2])^2)/(2*vt^2))
end

local function elc_maxwellian2v_truncated(v, vDrift, vt)
    if -3*vtElc < v[1]-vDrift[1] < 3*vtElc then 
       return 1/(2*math.pi*vt^2)*math.exp(-((v[1]-vDrift[1])^2+(v[2]-vDrift[2])^2)/(2*vt^2))
    else 
       return 0
    end
end

plasmaApp = Plasma.App {
    logToFile = true,
 
    tEnd        = 600,          -- End time. RLW: I changed this, but didn't change nFrame below.  So we might get more frames.
    nFrame      = 300,             -- Number of output frames.  This is 0.5 frames in unit time --> 3 frames every Langmuir period. 
    nDistFuncFrame = 100,           -- Number of distribution function output frames 

    lower       = {0.0,0.0},             -- Configuration space lower left.
    upper       = {1.0,1.0},             -- Configuration space upper right.
    cells       = {32,32},               -- Configuration space cells.
    basis       = "serendipity",    -- One of "serendipity" or "maximal-order".
    polyOrder   = 2,           -- Polynomial order.
    timeStepper = "rk3",            -- one of "rk2" or "rk3".
    cflFrac     = 0.9,
 
    -- Decomposition for configuration space.
    decompCuts = {16, 16},    -- Cuts in each configuration direction.
    useShared  = false,    -- If to use shared memory.
 
    -- Boundary conditions for configuration space.
    periodicDirs = {1,2}, -- Periodic directions.
    -- Integrated moment flag, compute quantities 1000 times in simulation.
    calcIntQuantEvery = 0.001,
 
    -- Electrons.
    elc = Plasma.Species {
       charge = -1.0, mass = 1.0,
       -- Velocity space grid.
       lower = {-3.0*vtElc,-3.0*vtElc},
       upper = {3.0*vtElc,3.0*vtElc},
       cells = {48,48},
       decompCuts = {1,1},
       -- initial conditions
       init = function (t, xn)
          local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
          local fv = maxwellian2v({vx, vy}, vDriftElc, vtElc)
          return fv 
       end,
       evolve = true, -- Evolve species?
 
       diagnosticMoments           = { "M0", "M1i" },
       diagnosticIntegratedMoments = { "intM0", "intM1i"},
    },
 
    -- Ions.
    ion = Plasma.Species {
       charge = 1.0, mass = massRatio,
       -- Velocity space grid.
       lower = {-6.0*vtIon,-6.0*vtIon},
       upper = {6.0*vtIon,6.0*vtIon},
       cells = {48,48},
       decompCuts = {1,1},
       -- Initial conditions.
       init = function (t, xn)
          local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
          local fv = maxwellian2v({vx, vy}, vDriftIon, vtIon)
          return fv*(1+perturbation*math.cos(2*math.pi*knumberx*x+ 2*math.pi*knumbery*y))    -- *(1.0 + perturbation*DrandKick({x,y},lx,nx,pOrder) ) RLW: I removed the noise
       end,
       evolve = true,    -- Evolve species?
 
       diagnosticMoments           = { "M0", "M1i", "M2" },
       diagnosticIntegratedMoments = { "intM0", "intM1i", "intM2Flow", "intM2Thermal", "intL2" },
    },

    -- Field solver.
    field = Plasma.Field {
       epsilon0 = 1.0,
       evolve = true, -- Evolve field?
       hasMagneticField = false,
    },
 
 }
 -- Run application.
 plasmaApp:run()
 
