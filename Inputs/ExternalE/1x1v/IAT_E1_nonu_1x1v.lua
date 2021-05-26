-- Gkyl -----------------------------------------------------------------------
-- Z.Liu 4/10/2021
-- This input file is a test for adding an external electric field, and a proper value for electric field.
-- Only one mode is excited. No noise. No collisions.
-- The result is that a reasonable value for E should be around 10^-5. Too large E will cause the electron 
-- to be accelerated to go outside of the velocity box

local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell()

-- Electron parameters.
vDriftElc = 0.00  -- Modified from 0.159.
vtElc     = 0.02
-- Ion parameters.
vDriftIon = 0.0
vtIon     = 0.000566      -- Modified from 0.001 (use Te = 50 Ti).
-- Mass ratio.
massRatio = 25.0  -- Modified from 25.

knumber      = 10.0    -- Wave-number.
perturbation = 1.0e-4  -- Distribution function perturbation.

local function maxwellian1v(v, vDrift, vt)
   return 1/math.sqrt(2*math.pi*vt^2)*math.exp(-(v-vDrift)^2/(2*vt^2))
end

plasmaApp = Plasma.App {
   logToFile = true,

   tEnd        = 1200.0,         -- End time.
   nFrame      = 600,           -- Number of output frames.

   lower       = {0.0},         -- Configuration space lower left.
   upper       = {1.0},         -- Configuration space upper right.
   cells       = {64},         -- Configuration space cells.
   basis       = "serendipity", -- One of "serendipity" or "maximal-order".
   polyOrder   = 2,             -- Polynomial order.
   timeStepper = "rk3",         -- One of "rk2" or "rk3".
   cflFrac     = 0.9,

   -- Eecomposition for configuration space
   decompCuts = {16},  -- Cuts in each configuration direction.
   useShared  = false, -- If to use shared memory.

   -- Boundary conditions for configuration space.
   periodicDirs = {1}, -- Periodic directions.
   -- Integrated moment flag, compute quantities 1000 times in simulation.
   calcIntQuantEvery = 0.001,   

   -- Electrons.
   elc = Plasma.Species {
      charge = -1.0, mass = 1.0,
      -- Velocity space grid.
      lower      = {-6.0},
      upper      = { 12.0},
      cells      = {108},
      decompCuts = {1},
      coordinateMap = {
         function (z)
            if z <= 10 then
               return (0.75*z-1.5)*vtElc
            else
               return (3*z-24)*vtElc
            end
         end
       },
      -- Initial conditions.
      init = function (t, xn)
         local x, v = xn[1], xn[2]
         local fv = maxwellian1v(v, vDriftElc, vtElc)
	 return fv*(1)
      end,
      evolve = true, -- Evolve species?

      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
   },

   -- Ions.
   ion = Plasma.Species {
      charge = 1.0, mass = massRatio,
      -- Velocity space grid.
      lower      = {-32.0*vtIon},   
      upper      = { 32.0*vtIon},
      cells      = {64},
      decompCuts = {1},
      -- Initial conditions.
      init = function (t, xn)
         local x, v = xn[1], xn[2]
         local fv = maxwellian1v(v, vDriftIon, vtIon)
         return fv*(1+perturbation*math.cos(2*math.pi*knumber*x))
      end,
      evolve = true, -- Evolve species?

      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
   },   

   -- Field solver.
   field = Plasma.Field {
      epsilon0 = 1.0,
      evolve   = true, -- Evolve field?
      hasMagneticField = false,
   },

   externalField = Plasma.ExternalField {
      hasMagneticField = false,
      emFunc = function(t, xn)
         local extE_x, extE_y, extE_z = -0.00001, 0., 0.
         return extE_x, extE_y, extE_z
      end,
      evolve = false, -- Evolve field?
   },
}
-- Run application.
plasmaApp:run()
