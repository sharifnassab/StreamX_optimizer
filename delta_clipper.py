import math, re


class delta_clipper:
    """
    clip_and_norm(delta) -> clipped_delta / normalizer

    Clipping (numbers parsed generically):
      'none'
      '1'   any float                     -> |delta| clipped to constant
      '10_avg_sq__dec_0.99'               -> |delta| <= 10 * avg, avg over delta^2
      '10_avg_abs__dec_0.95'              -> |delta| <= 10 * avg, avg over |delta|
      '10_avg_abs_max_1__dec_0.9'         -> trace uses min(|d|, 1)
      '10_avg_sq_max_20avg__dec_0.99'     -> trace uses min(|d|, 20*prev_avg)^2

    Normalization options (generic, case-insensitive):
      'none'
      '.99sq'        : EMA over delta^2  -> normalizer = sqrt(bias_corrected(EMA))
      '.995clipSq'   : EMA over clipped^2
      '.99abs'       : EMA over |delta|  -> normalizer = bias_corrected(EMA)
      '.995clipAbs'  : EMA over |clipped|
    """


    # -------- parsers (pure; return dicts) --------
    @staticmethod
    def parse_clip(clip_type: str):
        """
        Returns dict keys (all default None):
          cap_abs, cap_mult, avg_kind('abs'|'sq'|None),
          cap_constant, cap_kavg, eta_clip
        """
        out = dict(cap_abs=None, cap_mult=None, avg_kind=None,
                   cap_constant=None, cap_kavg=None, eta_clip=None)
        if not clip_type or clip_type == 'none':
            return out
        
        num = r'(?:\d+(?:\.\d+)?|\.\d+)'  # matches 1, 1.0, 0.99, .99

        m_abs = re.fullmatch(fr'({num})', clip_type)
        if m_abs:
            out['cap_abs'] = float(m_abs.group(1))
            return out

        m_mult = re.search(fr'^({num})_avg_', clip_type)
        if m_mult:
            out['cap_mult'] = float(m_mult.group(1))

        if '_avg_sq' in clip_type:
            out['avg_kind'] = 'sq'
        elif '_avg_abs' in clip_type:
            out['avg_kind'] = 'abs'

        m_c = re.search(fr'_max_({num})__', clip_type)
        if m_c:
            out['cap_constant'] = float(m_c.group(1))

        m_k = re.search(fr'_max_({num})avg__', clip_type)
        if m_k:
            out['cap_kavg'] = float(m_k.group(1))

        m_b = re.search(fr'__dec_({num})$', clip_type)
        if m_b:
            out['eta_clip'] = float(m_b.group(1))


        return out

    @staticmethod
    def parse_norm(normalization_type: str):
        """
        Accepts: 'none', '.99sq', '.995clipSq', '.99abs', '.995clipAbs'
        Returns dict keys (defaults None):
          eta_norm (float|None), norm_kind ('sq'|'abs'|None), use_clipped (bool|None)
        """
        out = dict(eta_norm=None, norm_kind=None, use_clipped=None)
        if not normalization_type or normalization_type.lower() == 'none':
            return out
        
        s = normalization_type.strip()
        m = re.match(r'^\.?(\d+(?:\.\d+)?)', s)
        if not m:
            raise ValueError("normalization_type must start with a float like '.99' or '0.99'")

        out['eta_norm'] = float(m.group(0))
        out['use_clipped'] = ('clip' in s)

        if 'Abs' in s:
            out['norm_kind'] = 'abs'
        elif 'Sq' in s:
            out['norm_kind'] = 'sq'
        else:
            raise ValueError("normalization_type must contain 'Abs' or 'Sq'")
        return out

    # ------------- init & state -------------
    def __init__(self, clip_type='none', normalization_type='none'):
        c = self.parse_clip(clip_type)
        n = self.parse_norm(normalization_type)

        # clip config
        self.cap_abs       = c['cap_abs']
        self.cap_mult      = c['cap_mult']
        self.avg_kind      = c['avg_kind']
        self.cap_constant  = c['cap_constant']
        self.cap_kavg      = c['cap_kavg']
        self.eta_clip      = c['eta_clip']        # decay for avg trace

        # norm config
        self.eta_norm      = n['eta_norm']        # EMA step
        self.norm_kind     = n['norm_kind']       # 'sq' or 'abs'
        self.use_clipped   = n['use_clipped']     # True for clip*, else None

        # states
        self.t_clip = 0
        self.moment = 0.0                         # EMA over |d| or d^2 (for clipping)
        self.t_norm = 0
        self.norm_state = 0.0                     # EMA over |d| (abs) or d^2 (sq)
        self._eps = 1e-12

        print(clip_type, normalization_type)
        print()
        print(c)
        print()
        print(n)
        print()
    # ------------- public -------------
    def clip_and_norm(self, delta: float) -> float:
        prev_avg = self._bias_corrected_avg()
        clipped  = self._clip_and_update(delta, prev_avg)
        denom    = self._norm_and_update(delta, clipped)
        delta_out = clipped / max([denom, self._eps])
        
        #if abs(delta)>2:
        #    print('\t'.join(format(x, '.3f').rstrip('0').rstrip('.') for x in (delta, delta_out, denom)))

        return delta_out

    # ------------- internals -------------
    def _bias_corrected_avg(self):
        if not self.avg_kind or self.t_clip == 0:
            return 0.0
        eta_clip = self.eta_clip if (self.eta_clip is not None) else 0.99
        denom = 1.0 - (eta_clip ** self.t_clip)
        if denom <= 0.0: return 0.0
        if self.avg_kind == 'sq':
            v = self.moment / denom
            return math.sqrt(max(v, 0.0))
        return self.moment / denom  # 'abs'

    def _clip_and_update(self, delta, prev_avg):
        a = abs(delta)

        # constant-absolute cap (no averaging)
        if self.cap_abs is not None and self.avg_kind is None:
            return math.copysign(min(a, self.cap_abs), delta)

        # no clipping at all
        if self.avg_kind is None and self.cap_abs is None:
            return float(delta)

        # trace for EMA (possibly capped)
        trace_abs = a
        if self.cap_constant is not None:
            trace_abs = min(trace_abs, self.cap_constant)
        if self.cap_kavg is not None and prev_avg > 0.0:
            trace_abs = min(trace_abs, self.cap_kavg * prev_avg)

        x = trace_abs * trace_abs if self.avg_kind == 'sq' else trace_abs

        # EMA update on trace
        self.t_clip += 1
        eta_clip = 0.99 if (self.eta_clip is None) else self.eta_clip
        self.moment += (1.0 - eta_clip) * (x - self.moment)

        # compute current threshold and clip
        curr_avg = self._bias_corrected_avg()
        cap = float('inf')
        if self.cap_mult is not None and curr_avg > 0:
            cap = self.cap_mult * curr_avg
        return math.copysign(min(a, cap), delta)

    def _norm_and_update(self, delta, clipped):
        # no normalization
        if self.eta_norm is None or self.norm_kind is None:
            return 1.0

        self.t_norm += 1
        # choose source (raw vs clipped)
        src = clipped if self.use_clipped else delta

        # update EMA
        if self.norm_kind == 'sq':
            x = src * src
        else:  # 'abs'
            x = abs(src)

        self.norm_state = self.eta_norm * self.norm_state  +   (1-self.eta_norm) * x 

        # bias correction
        denom = 1.0 - self.eta_norm ** self.t_norm
        if denom <= 0.0:
            return 1.0
        hat = self.norm_state / denom

        # map to normalizer
        if self.norm_kind == 'sq':
            return math.sqrt(max(hat, 0.0))
        else:  # 'abs'
            return max(hat, 0.0)
