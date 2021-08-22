// **********************************************************************
//
// Copyright (c) 2003-2017 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************
//
// Ice version 3.7.0
//
// <auto-generated>
//
// Generated from file `ObjectDetector.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#ifndef __ObjectDetector_h__
#define __ObjectDetector_h__

#include <IceUtil/PushDisableWarnings.h>
#include <Ice/ProxyF.h>
#include <Ice/ObjectF.h>
#include <Ice/ValueF.h>
#include <Ice/Exception.h>
#include <Ice/LocalObject.h>
#include <Ice/StreamHelpers.h>
#include <Ice/Comparable.h>
#include <Ice/Proxy.h>
#include <Ice/Object.h>
#include <Ice/GCObject.h>
#include <Ice/Value.h>
#include <Ice/Incoming.h>
#include <Ice/FactoryTableInit.h>
#include <IceUtil/ScopedArray.h>
#include <Ice/Optional.h>
#include <IceUtil/UndefSysMacros.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 307
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 0
#       error Ice patch level mismatch!
#   endif
#endif

#ifdef ICE_CPP11_MAPPING // C++11 mapping

namespace RoboCompObjectDetector
{

class ObjectDetector;
class ObjectDetectorPrx;

}

namespace RoboCompObjectDetector
{

struct ObjectT
{
    float timestamp;
    int id;
    float x;
    float y;
    float angle;
    float ix;
    float iy;
    float iangle;
    float bbx1;
    float bby1;
    float bbx2;
    float bby2;
    bool collision;

    std::tuple<const float&, const int&, const float&, const float&, const float&, const float&, const float&, const float&, const float&, const float&, const float&, const float&, const bool&> ice_tuple() const
    {
        return std::tie(timestamp, id, x, y, angle, ix, iy, iangle, bbx1, bby1, bbx2, bby2, collision);
    }
};

using ObjectList = ::std::vector<::RoboCompObjectDetector::ObjectT>;

using Ice::operator<;
using Ice::operator<=;
using Ice::operator>;
using Ice::operator>=;
using Ice::operator==;
using Ice::operator!=;

}

namespace RoboCompObjectDetector
{

class ObjectDetector : public virtual ::Ice::Object
{
public:

    using ProxyType = ObjectDetectorPrx;

    virtual bool ice_isA(::std::string, const ::Ice::Current&) const override;
    virtual ::std::vector<::std::string> ice_ids(const ::Ice::Current&) const override;
    virtual ::std::string ice_id(const ::Ice::Current&) const override;

    static const ::std::string& ice_staticId();

    virtual void gotobjects(::RoboCompObjectDetector::ObjectList, const ::Ice::Current&) = 0;
    bool _iceD_gotobjects(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&) override;
};

}

namespace RoboCompObjectDetector
{

class ObjectDetectorPrx : public virtual ::Ice::Proxy<ObjectDetectorPrx, ::Ice::ObjectPrx>
{
public:

    void gotobjects(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::Ice::Context& context = Ice::noExplicitContext)
    {
        _makePromiseOutgoing<void>(true, this, &RoboCompObjectDetector::ObjectDetectorPrx::_iceI_gotobjects, iceP_lst, context).get();
    }

    template<template<typename> class P = ::std::promise>
    auto gotobjectsAsync(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::Ice::Context& context = Ice::noExplicitContext)
        -> decltype(::std::declval<P<void>>().get_future())
    {
        return _makePromiseOutgoing<void, P>(false, this, &RoboCompObjectDetector::ObjectDetectorPrx::_iceI_gotobjects, iceP_lst, context);
    }

    ::std::function<void()>
    gotobjectsAsync(const ::RoboCompObjectDetector::ObjectList& iceP_lst,
                    ::std::function<void()> response,
                    ::std::function<void(::std::exception_ptr)> ex = nullptr,
                    ::std::function<void(bool)> sent = nullptr,
                    const ::Ice::Context& context = Ice::noExplicitContext)
    {
        return _makeLamdaOutgoing<void>(response, ex, sent, this, &RoboCompObjectDetector::ObjectDetectorPrx::_iceI_gotobjects, iceP_lst, context);
    }

    void _iceI_gotobjects(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>&, const ::RoboCompObjectDetector::ObjectList&, const ::Ice::Context&);

    static const ::std::string& ice_staticId();

protected:

    ObjectDetectorPrx() = default;
    friend ::std::shared_ptr<ObjectDetectorPrx> IceInternal::createProxy<ObjectDetectorPrx>();

    virtual ::std::shared_ptr<::Ice::ObjectPrx> _newInstance() const override;
};

}

namespace Ice
{

template<>
struct StreamableTraits<::RoboCompObjectDetector::ObjectT>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 49;
    static const bool fixedLength = true;
};

template<typename S>
struct StreamReader<::RoboCompObjectDetector::ObjectT, S>
{
    static void read(S* istr, ::RoboCompObjectDetector::ObjectT& v)
    {
        istr->readAll(v.timestamp, v.id, v.x, v.y, v.angle, v.ix, v.iy, v.iangle, v.bbx1, v.bby1, v.bbx2, v.bby2, v.collision);
    }
};

}

namespace RoboCompObjectDetector
{

using ObjectDetectorPtr = ::std::shared_ptr<ObjectDetector>;
using ObjectDetectorPrxPtr = ::std::shared_ptr<ObjectDetectorPrx>;

}

#else // C++98 mapping

namespace IceProxy
{

namespace RoboCompObjectDetector
{

class ObjectDetector;
void _readProxy(::Ice::InputStream*, ::IceInternal::ProxyHandle< ::IceProxy::RoboCompObjectDetector::ObjectDetector>&);
::IceProxy::Ice::Object* upCast(::IceProxy::RoboCompObjectDetector::ObjectDetector*);

}

}

namespace RoboCompObjectDetector
{

class ObjectDetector;
::Ice::Object* upCast(::RoboCompObjectDetector::ObjectDetector*);
typedef ::IceInternal::Handle< ::RoboCompObjectDetector::ObjectDetector> ObjectDetectorPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::RoboCompObjectDetector::ObjectDetector> ObjectDetectorPrx;
typedef ObjectDetectorPrx ObjectDetectorPrxPtr;
void _icePatchObjectPtr(ObjectDetectorPtr&, const ::Ice::ObjectPtr&);

}

namespace RoboCompObjectDetector
{

struct ObjectT
{
    ::Ice::Float timestamp;
    ::Ice::Int id;
    ::Ice::Float x;
    ::Ice::Float y;
    ::Ice::Float angle;
    ::Ice::Float ix;
    ::Ice::Float iy;
    ::Ice::Float iangle;
    ::Ice::Float bbx1;
    ::Ice::Float bby1;
    ::Ice::Float bbx2;
    ::Ice::Float bby2;
    bool collision;
};

typedef ::std::vector< ::RoboCompObjectDetector::ObjectT> ObjectList;

}

namespace RoboCompObjectDetector
{

class Callback_ObjectDetector_gotobjects_Base : public virtual ::IceInternal::CallbackBase { };
typedef ::IceUtil::Handle< Callback_ObjectDetector_gotobjects_Base> Callback_ObjectDetector_gotobjectsPtr;

}

namespace IceProxy
{

namespace RoboCompObjectDetector
{

class ObjectDetector : public virtual ::Ice::Proxy<ObjectDetector, ::IceProxy::Ice::Object>
{
public:

    void gotobjects(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        end_gotobjects(_iceI_begin_gotobjects(iceP_lst, context, ::IceInternal::dummyCallback, 0, true));
    }

    ::Ice::AsyncResultPtr begin_gotobjects(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        return _iceI_begin_gotobjects(iceP_lst, context, ::IceInternal::dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_gotobjects(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotobjects(iceP_lst, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotobjects(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::Ice::Context& context, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotobjects(iceP_lst, context, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotobjects(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::RoboCompObjectDetector::Callback_ObjectDetector_gotobjectsPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotobjects(iceP_lst, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotobjects(const ::RoboCompObjectDetector::ObjectList& iceP_lst, const ::Ice::Context& context, const ::RoboCompObjectDetector::Callback_ObjectDetector_gotobjectsPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotobjects(iceP_lst, context, del, cookie);
    }

    void end_gotobjects(const ::Ice::AsyncResultPtr&);

private:

    ::Ice::AsyncResultPtr _iceI_begin_gotobjects(const ::RoboCompObjectDetector::ObjectList&, const ::Ice::Context&, const ::IceInternal::CallbackBasePtr&, const ::Ice::LocalObjectPtr& cookie = 0, bool sync = false);

public:

    static const ::std::string& ice_staticId();

protected:

    virtual ::IceProxy::Ice::Object* _newInstance() const;
};

}

}

namespace RoboCompObjectDetector
{

class ObjectDetector : public virtual ::Ice::Object
{
public:

    typedef ObjectDetectorPrx ProxyType;
    typedef ObjectDetectorPtr PointerType;

    virtual ~ObjectDetector();

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::emptyCurrent) const;

    static const ::std::string& ice_staticId();

    virtual void gotobjects(const ::RoboCompObjectDetector::ObjectList&, const ::Ice::Current& = ::Ice::emptyCurrent) = 0;
    bool _iceD_gotobjects(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&);

protected:

    virtual void _iceWriteImpl(::Ice::OutputStream*) const;
    virtual void _iceReadImpl(::Ice::InputStream*);
};

inline bool operator==(const ObjectDetector& lhs, const ObjectDetector& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) == static_cast<const ::Ice::Object&>(rhs);
}

inline bool operator<(const ObjectDetector& lhs, const ObjectDetector& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) < static_cast<const ::Ice::Object&>(rhs);
}

}

namespace Ice
{

template<>
struct StreamableTraits< ::RoboCompObjectDetector::ObjectT>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 49;
    static const bool fixedLength = true;
};

template<typename S>
struct StreamWriter< ::RoboCompObjectDetector::ObjectT, S>
{
    static void write(S* ostr, const ::RoboCompObjectDetector::ObjectT& v)
    {
        ostr->write(v.timestamp);
        ostr->write(v.id);
        ostr->write(v.x);
        ostr->write(v.y);
        ostr->write(v.angle);
        ostr->write(v.ix);
        ostr->write(v.iy);
        ostr->write(v.iangle);
        ostr->write(v.bbx1);
        ostr->write(v.bby1);
        ostr->write(v.bbx2);
        ostr->write(v.bby2);
        ostr->write(v.collision);
    }
};

template<typename S>
struct StreamReader< ::RoboCompObjectDetector::ObjectT, S>
{
    static void read(S* istr, ::RoboCompObjectDetector::ObjectT& v)
    {
        istr->read(v.timestamp);
        istr->read(v.id);
        istr->read(v.x);
        istr->read(v.y);
        istr->read(v.angle);
        istr->read(v.ix);
        istr->read(v.iy);
        istr->read(v.iangle);
        istr->read(v.bbx1);
        istr->read(v.bby1);
        istr->read(v.bbx2);
        istr->read(v.bby2);
        istr->read(v.collision);
    }
};

}

namespace RoboCompObjectDetector
{

template<class T>
class CallbackNC_ObjectDetector_gotobjects : public Callback_ObjectDetector_gotobjects_Base, public ::IceInternal::OnewayCallbackNC<T>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception&);
    typedef void (T::*Sent)(bool);
    typedef void (T::*Response)();

    CallbackNC_ObjectDetector_gotobjects(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallbackNC<T>(obj, cb, excb, sentcb)
    {
    }
};

template<class T> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(const IceUtil::Handle<T>& instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ObjectDetector_gotobjects<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ObjectDetector_gotobjects<T>(instance, 0, excb, sentcb);
}

template<class T> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(T* instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ObjectDetector_gotobjects<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(T* instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ObjectDetector_gotobjects<T>(instance, 0, excb, sentcb);
}

template<class T, typename CT>
class Callback_ObjectDetector_gotobjects : public Callback_ObjectDetector_gotobjects_Base, public ::IceInternal::OnewayCallback<T, CT>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception& , const CT&);
    typedef void (T::*Sent)(bool , const CT&);
    typedef void (T::*Response)(const CT&);

    Callback_ObjectDetector_gotobjects(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallback<T, CT>(obj, cb, excb, sentcb)
    {
    }
};

template<class T, typename CT> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(const IceUtil::Handle<T>& instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ObjectDetector_gotobjects<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ObjectDetector_gotobjects<T, CT>(instance, 0, excb, sentcb);
}

template<class T, typename CT> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(T* instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ObjectDetector_gotobjects<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_ObjectDetector_gotobjectsPtr
newCallback_ObjectDetector_gotobjects(T* instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ObjectDetector_gotobjects<T, CT>(instance, 0, excb, sentcb);
}

}

#endif

#include <IceUtil/PopDisableWarnings.h>
#endif
