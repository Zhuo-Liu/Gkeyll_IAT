-- Gkyl -----------------------------------------------------------------------
-- Z.Liu 2/23/2022
-- This input file is for linear benchmarking of Ion acoustic instability

local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell()

-- electron parameters
vDriftElc = 0.0339   ---modified from 0.159
vtElc = 0.02
-- ion parameters
vDriftIon = 0.0
vtIon = 0.002828      ---- modified from 0.001 (use Te = 50 Ti)
-- mass ratio
massRatio = 25.0  ----modified from 25 

knumber = 10.0 -- wave-number
perturbation = 1.0e-4 -- distribution function perturbation

local function maxwellian1v(v, vDrift, vt)
   return 1/math.sqrt(2*math.pi*vt^2)*math.exp(-(v-vDrift)^2/(2*vt^2))
end

plasmaApp = Plasma.App {
   logToFile = true,

   tEnd = 600.0, -- end time
   nFrame = 300, -- number of output frames

   lower = {0.0}, -- configuration space lower left
   upper = {1.0}, -- configuration space upper right
   cells = {256}, -- configuration space cells
   basis = "serendipity", -- one of "serendipity" or "maximal-order"
   polyOrder = 2, -- polynomial order
   timeStepper = "rk3", -- one of "rk2" or "rk3"
   cflFrac = 0.9,

   -- decomposition for configuration space
   decompCuts = {32}, -- cuts in each configuration direction
   useShared = false, -- if to use shared memory

   -- boundary conditions for configuration space
   periodicDirs = {1}, -- periodic directions
   -- integrated moment flag, compute quantities 1000 times in simulation
   calcIntQuantEvery = 0.001,   

   -- electrons
   elc = Plasma.Species {
      charge = -1.0, mass = 1.0,
      -- velocity space grid
      lower = {-8.0*vtElc},
      upper = {8.0*vtElc},
      cells = {256},
      decompCuts = {1},
      -- initial conditions
      init = function (t, xn)
       local x, v = xn[1], xn[2]
        local fv = maxwellian1v(v, vDriftElc, vtElc)
	 return fv*(1)
      end,
      evolve = true, -- evolve species?

      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
   },

   -- ions
   ion = Plasma.Species {
      charge = 1.0, mass = massRatio,
      -- velocity space grid
      lower = {-20.0*vtIon},   
      upper = {20.0*vtIon},
      cells = {128},
      decompCuts = {1},
      -- initial conditions
      init = function (t, xn)
       local x, v = xn[1], xn[2]
        local fv = maxwellian1v(v, vDriftIon, vtIon)
	  return fv*(1+perturbation*math.cos(2*math.pi*knumber*x))
      end,
      evolve = true, -- evolve species?

      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
   },   

   -- field solver
   field = Plasma.Field {
      epsilon0 = 1.0,
      evolve = true, -- evolve field?
      hasMagneticField = false,
   },
}
-- run application
plasmaApp:run()
