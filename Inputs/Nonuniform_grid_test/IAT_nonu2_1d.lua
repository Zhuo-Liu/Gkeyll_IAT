-- Gkyl -----------------------------------------------------------------------
-- Z.Liu 4/27/2021
-- This input file is intended to expand the simulation box in velocity space 4 times as before
local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell()

-- electron parameters
vDriftElc = 0.01   ---modified from 0.159
vtElc = 0.02
-- ion parameters
vDriftIon = 0.0
vtIon = 0.000566      ---- modified from 0.001 (use Te = 50 Ti)
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
   cells = {32}, -- configuration space cells
   basis = "serendipity", -- one of "serendipity" or "maximal-order"
   polyOrder = 2, -- polynomial order
   timeStepper = "rk3", -- one of "rk2" or "rk3"
   cflFrac = 0.9,

   -- decomposition for configuration space
   decompCuts = {16}, -- cuts in each configuration direction
   useShared = false, -- if to use shared memory

   -- boundary conditions for configuration space
   periodicDirs = {1}, -- periodic directions
   -- integrated moment flag, compute quantities 1000 times in simulation
   calcIntQuantEvery = 0.001,   

   -- electrons
   elc = Plasma.Species {
      charge = -1.0, mass = 1.0,
      -- velocity space grid
      lower = {-12.0},
      upper = {12.0},
      cells = {96},
      decompCuts = {1},
      coordinateMap = {
        function (z)
            if z >= -6 and z <= 6 then
                return 0.5*z*vtElc
            elseif z>6 then
                return (1.5*z-6)*vtElc
            else
                return (1.5*z+6)*vtElc
            end
        end
      },
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
      lower = {-16.0*vtIon},   
      upper = {16.0*vtIon},
      cells = {48},
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
