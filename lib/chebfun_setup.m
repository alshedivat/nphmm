function chebfun_setup()
  % Setup Chebfun preferences
  chebfunpref.setDefaults('splitting', 1);
  %chebfunpref.setDefaults('maxLength', 32769);
  chebfunpref.setDefaults('chebfuneps', 1e-10);
  chebfunpref.setDefaults({'cheb2Prefs', 'chebfun2eps'}, 1e-10);
  chebfunpref.setDefaults({'cheb3Prefs', 'chebfun3eps'}, 1e-4);
end
