-- Gkyl ------------------------------------------------------------------------
local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell()

-- electron parameters
vDriftElc = {0.01,0}   ---modified from 0.159
vtElc = 0.02
-- ion parameters
vDriftIon = {0.0,0.0}
vtIon = 0.000566      ---- modified from 0.001 (use Te = 50 Ti)
tempRatio = 50.0
-- mass ratio
massRatio = 25.0  ----modified from 25 

knumberx = 10.0 -- wave-number
knumbery = 0.0
perturbation = 1.0e-4 -- distribution function perturbation

omegaPe = 1.0
omegaPi = omegaPe/math.sqrt(massRatio)
lambdaDe = vtElc / omegaPe

-- Collision frequencies.
nuee = 0.0005
nuei = 0.0 --nuee              --RLW 
nuii = nuee/math.sqrt(massRatio)  --should have been nuee/math.sqrt(massRatio)*math.pow(tempRatio,1.5)
nuie = 0.0 -- nuee/massRatio

lx = {1.0,0.5}
nx = {32,16}

pOrder = 2

local function maxwellian2v(v, vDrift, vt)
    return 1/(2*math.pi*vt^2)*math.exp(-((v[1]-vDrift[1])^2+(v[2]-vDrift[2])^2)/(2*vt^2))
end

local function sponEmissionSource(x_table, t, lx_table, ncells_table, p)
    -- x_table = {x, y} are independent variables.
    -- t is the time
    -- lx_table = {xmax-xmin, ymax - ymin}
    -- ncells_table = {ncellsx, ncellsy}
    -- p is polynomial order.
    ------------------------
    -- returns stochastic source term (\delta E_x, \delta E_y, delta f_i, \delta f_e) that models spontaneous emission of ion-acoustic waves
    local Nx = ncells_table[1]*(p+1) -- Number of spatial degrees of freedom along x
    local Ny = ncells_table[2]*(p+1) -- Number of spatial degrees of freedom along y
    local Lx, Ly = lx_table[1], lx_table[2]
    local x, y, vx, vy = x_table[1], x_table[2], x_table[3], x_table[4]
    local omega = 0.0 
    local modE = 0.0             -- Initialize |J|.
    local kdotv_omega = 0.0   -- k \cdot v / omega
    local fIon, fElc, fkIon, fkElc = 0.0, 0.0, 0.0, 0.0
    local ksquared = 0.0
    math.randomseed(math.floor(1000000*t)) --I want all xs and ys (and vx, vy) to see the same random phase at a given time step.  Since t will be a fraction, need to multiply it so that we don't get the same random number thousands of times.
    for nx = -math.floor(Nx/2), math.floor(Nx/2) do   --need to divied by two because we are including nx and -nx by using sines and cosines
       for ny = -math.floor(Ny/2), math.floor(Ny/2) do
         -- kx, ky = {2 pi nx/Lx, 2 pi ny/Ly}
         ksquared = math.pow(2*nx*math.pi /Lx,2) + math.pow(2*ny*math.pi/Ly,2)
         if ksquared > 0.0 then
            omega = omegaPi / math.sqrt(1.0 + 1.0/(ksquared*lambdaDe^2))
            kdotv_omega =(2*math.pi*nx*vx/Lx+ 2*math.pi*ny*vy/Ly)/omega 
            modE = 0.141421*math.pow(math.random(),2)	--magnitude of J for these mode numbers
            fkIon = tempRatio/(ksquared*math.pow(lambdaDe,2))*(math.sqrt(ksquared)*modE)*(kdotv_omega + math.pow(kdotv_omega,2))*math.cos(2*math.pi*(nx*x/Lx + ny*y/Ly  + math.random() ))
            fkElc = -1.0/(ksquared*math.pow(lambdaDe,2))* (math.sqrt(ksquared)*modE) * (-1.0 )*math.cos(2*math.pi*(nx*x/Lx + ny*y/Ly  + math.random() ))  --one negative sign is from the electron charge; the other is from the simplified form of v/(omega - k.v)
            fIon = fIon + fkIon
            fElc = fElc + fkElc
          end  
       end
    end
    return {fIon,fElc}
 end

plasmaApp = Plasma.App {
    logToFile = true,
 
    tEnd        = 600,          -- End time. RLW: I changed this, but didn't change nFrame below.  So we might get more frames.
    nFrame      = 300,             -- Number of output frames.  This is 0.5 frames in unit time --> 3 frames every Langmuir period. 
    nDistFuncFrame = 100,           -- Number of distribution function output frames 

    lower       = {0.0,0.0},             -- Configuration space lower left.
    upper       = {1.0,0.5},             -- Configuration space upper right.
    cells       = {32,16},               -- Configuration space cells.
    basis       = "serendipity",    -- One of "serendipity" or "maximal-order".
    polyOrder   = pOrder,           -- Polynomial order.
    timeStepper = "rk3",            -- one of "rk2" or "rk3".
    cflFrac     = 0.9,
 
    -- Decomposition for configuration space.
    decompCuts = {16, 8},    -- Cuts in each configuration direction.
    useShared  = false,    -- If to use shared memory.
 
    -- Boundary conditions for configuration space.
    periodicDirs = {1,2}, -- Periodic directions.
    -- Integrated moment flag, compute quantities 1000 times in simulation.
    calcIntQuantEvery = 0.001,
 
    -- Electrons.
    elc = Plasma.Species {
       charge = -1.0, mass = 1.0,
       -- Velocity space grid.
       lower = {-6.0*vtElc,-3.0*vtElc},
       upper = {6.0*vtElc,3.0*vtElc},
       cells = {96,48},
       decompCuts = {1,1},
       -- initial conditions
       init = function (t, xn)
          local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
          local fv = maxwellian2v({vx, vy}, vDriftElc, vtElc)
          local dfv = sponEmissionSource({x,y,vx,vy},t, lx, nx, pOrder)[2]*maxwellian2v({vx, vy}, vDriftElc, vtElc)
          return fv + perturbation*dfv
       end,
       evolve = true, -- Evolve species?
 
       diagnosticMoments           = { "M0", "M1i" },
       diagnosticIntegratedMoments = { "intM0", "intM1i"},
       coll = Plasma.LBOCollisions{
         collideWith = {'elc', 'ion'},
	 frequencies = {nuee, nuei},
      }
    },
 
    -- Ions.
    ion = Plasma.Species {
       charge = 1.0, mass = massRatio,
       -- Velocity space grid.
       lower = {-12.0*vtIon,-6.0*vtIon},
       upper = {12.0*vtIon,6.0*vtIon},
       cells = {32,16},
       decompCuts = {1,1},
       -- Initial conditions.
       init = function (t, xn)
          local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
          local fv = maxwellian2v({vx, vy}, vDriftIon, vtIon)
          local dfv = sponEmissionSource({x,y,vx,vy},t, lx, nx, pOrder)[1]*maxwellian2v({vx,vy},vDriftIon, vtIon) 
          return fv + perturbation*dfv    -- *(1.0 + perturbation*DrandKick({x,y},lx,nx,pOrder) ) RLW: I removed the noise
       end,
       evolve = true,    -- Evolve species?
 
       diagnosticMoments           = { "M0", "M1i", "M2" },
       diagnosticIntegratedMoments = { "intM0", "intM1i", "intM2Flow", "intM2Thermal", "intL2" },
       coll = Plasma.LBOCollisions{
         collideWith = {'elc', 'ion'},
	 frequencies = {nuie, nuii},
      }
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
 
