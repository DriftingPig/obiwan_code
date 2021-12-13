/* -*- c++ -*-
 * Copyright (c) 2012-2019 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef GalSim_SBTransformImpl_H
#define GalSim_SBTransformImpl_H

#include "SBProfileImpl.h"
#include "SBTransform.h"

namespace galsim {

    class SBTransform::SBTransformImpl : public SBProfileImpl
    {
    public:

        SBTransformImpl(const SBProfile& sbin, double mA, double mB, double mC, double mD,
                        const Position<double>& cen, double ampScaling, const GSParams& gsparams);

        ~SBTransformImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return _stillIsAxisymmetric; }
        bool hasHardEdges() const { return _adaptee.hasHardEdges(); }
        bool isAnalyticX() const { return _adaptee.isAnalyticX(); }
        bool isAnalyticK() const { return _adaptee.isAnalyticK(); }

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const;

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const;

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const;

        Position<double> centroid() const { return _cen + fwd(_adaptee.centroid()); }

        double getFlux() const { return _adaptee.getFlux() * _fluxScaling; }
        double maxSB() const { return _adaptee.maxSB() * _ampScaling; }

        double getPositiveFlux() const { return _adaptee.getPositiveFlux() * _fluxScaling; }
        double getNegativeFlux() const { return _adaptee.getNegativeFlux() * _fluxScaling; }

        /**
         * @brief Shoot photons through this SBTransform.
         *
         * SBTransform will simply apply the affine transformation to coordinates of photons
         * generated by its adaptee, and rescale the flux by the determinant of the distortion
         * matrix.
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        SBProfile getObj() const { return _adaptee; }
        void getJac(double& mA, double& mB, double& mC, double& mD) const
        { mA = _mA; mB = _mB; mC = _mC; mD = _mD; }
        Position<double> getOffset() const { return _cen; }
        double getFluxScaling() const { return _ampScaling; }

        // Overrides for better efficiency
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const;
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

        std::string serialize() const;

    private:
        SBProfile _adaptee; ///< SBProfile being adapted/transformed

        double _mA; ///< A element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        double _mB; ///< B element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        double _mC; ///< C element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        double _mD; ///< D element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        Position<double> _cen;  ///< Centroid position.

        // Calculate and save these:
        double _absdet;  ///< Determinant (flux magnification) of `M` matrix * ampScaling
        double _ampScaling;  ///< Amount to scale amplitude by
        double _fluxScaling;  ///< Amount to scale flux by (= absdet * ampScaling)
        double _invdet;  ///< Inverse determinant of `M` matrix.
        bool _stillIsAxisymmetric; ///< Is output SBProfile shape still circular?
        bool _zeroCen;
        double _major, _minor;

        mutable double _maxk;
        mutable double _stepk;
        mutable double _xmin, _xmax, _ymin, _ymax; ///< Ranges propagated from adaptee
        mutable double _coeff_b, _coeff_c, _coeff_c2; ///< Values used in getYRangeX(x,ymin,ymax);
        mutable std::vector<double> _xsplits, _ysplits; ///< Good split points for the intetegrals

        void setupRanges() const;

        /**
         * @brief Forward coordinate transform with `M` matrix.
         *
         * @param[in] p input position.
         * @returns transformed position.
         */
        Position<double> fwd(const Position<double>& p) const
        { return _fwd(_mA,_mB,_mC,_mD,p.x,p.y,_invdet); }

        /// @brief Forward coordinate transform with transpose of `M` matrix.
        Position<double> fwdT(const Position<double>& p) const
        { return _fwd(_mA,_mC,_mB,_mD,p.x,p.y,_invdet); }

        /// @brief Inverse coordinate transform with `M` matrix.
        Position<double> inv(const Position<double>& p) const
        { return _inv(_mA,_mB,_mC,_mD,p.x,p.y,_invdet); }

        /// @brief Returns the k value (no phase).
        std::complex<double> kValueNoPhase(const Position<double>& k) const;

        std::complex<double> (*_kValue)(
            const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
            const Position<double>& k, const Position<double>& cen);
        std::complex<double> (*_kValueNoPhase)(
            const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
            const Position<double>& , const Position<double>& );

        Position<double> (*_fwd)(
            double mA, double mB, double mC, double mD, double x, double y, double );
        Position<double> (*_inv)(
            double mA, double mB, double mC, double mD, double x, double y, double invdet);

        static std::complex<double> _kValueNoPhaseNoDet(
            const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
            const Position<double>& , const Position<double>& );
        static std::complex<double> _kValueNoPhaseWithDet(
            const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
            const Position<double>& , const Position<double>& );
        static std::complex<double> _kValueWithPhase(
            const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
            const Position<double>& k, const Position<double>& cen);

        static Position<double> _fwd_normal(
            double mA, double mB, double mC, double mD, double x, double y, double )
        { return Position<double>(mA*x + mB*y, mC*x + mD*y); }
        static Position<double> _inv_normal(
            double mA, double mB, double mC, double mD, double x, double y, double invdet)
        { return Position<double>(invdet*(mD*x - mB*y), invdet*(-mC*x + mA*y)); }
        static Position<double> _ident(
            double , double , double , double , double x, double y, double )
        { return Position<double>(x,y); }

        void doFillXImage(ImageView<double> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<double> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }

        // Copy constructor and op= are undefined.
        SBTransformImpl(const SBTransformImpl& rhs);
        void operator=(const SBTransformImpl& rhs);
    };
}

#endif