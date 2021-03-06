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
knumbery = 0.0
perturbation = 1.0e-4 -- distribution function perturbation

local function maxwellian2v(v, vDrift, vt)
    return 1/(2*math.pi*vt^2)*math.exp(-((v[1]-vDrift[1])^2+(v[2]-vDrift[2])^2)/(2*vt^2))
end

plasmaApp = Plasma.App {
    logToFile = true,
 
    tEnd        = 300,          -- End time. RLW: I changed this, but didn't change nFrame below.  So we might get more frames.
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
       lower = {-6.0*vtElc,-6.0*vtElc},
       upper = {6.0*vtElc,6.0*vtElc},
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
       epsilon0 = 1.0, mu0 = 1.0,
       --useGhostCurrent = true,
       init = function (t, xn)
          local ksquared =  math.pow(2*knumberx*math.pi,2) + math.pow(2*knumbery*math.pi,2)
          local x, y = xn[1], xn[2]
          local Ex = perturbation*math.sin(2*math.pi*knumberx*x+ 2*math.pi*knumbery*y)*knumberx/math.sqrt(ksquared)
          local Ey = perturbation*math.sin(2*math.pi*knumberx*x+ 2*math.pi*knumbery*y)*knumbery/math.sqrt(ksquared)
          return Ex,Ey, 0.0, 0.0, 0.0, 0.0
       end,
       evolve = true, -- Evolve field?
    },
 
    -- Current antenna.
    driveSpecies = Plasma.FuncSpecies {
       charge = 1.0, mass = 1.0,
       momentumDensity = function (t, xn)
          local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
          local Jx = (vDriftElc[1] - vDriftIon[1])
          local Jy = (vDriftElc[2] - vDriftIon[2])
          --local dJx =  sponEmissionSource({x,y,0.0,0.0},t, lx, nx, pOrder)[1]
          --local dJy =  sponEmissionSource({x,y,0.0,0.0},t, lx, nx, pOrder)[2]
          -- local dJx, dJy =  FluctuatingCurrentDrive({x, y}, t, lx, nx, pOrder)
          --return Jx + noise*dJx, noise*dJy
          return Jx,Jy
       end,
       evolve = true, -- Evolve field?
    },
 }
 -- Run application.
 plasmaApp:run()
 